"""Input model and JSON loading.

The on-disk format is a single JSON object describing one account and its posts.
Only ``posts[].timestamp`` and ``posts[].text`` are strictly required; every other
field is optional and missing data degrades a signal to ``None`` rather than being
guessed. See the README "Input format" section for the documented schema.
"""

from __future__ import annotations

import json
import math
import os
import stat
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# Input is collected from accounts that may be adversarial, so the loader treats
# every field as hostile: sizes are bounded, types are checked, and anything that
# reaches the terminal is stripped of control characters.
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_POSTS = 200_000
MAX_HANDLE_CHARS = 64
MAX_COUNT = 10**12  # ceiling for likes/reposts/followers
# Timestamps outside this range are not plausible post times, and extreme dates
# overflow datetime arithmetic in the rolling-window signals.
_MIN_TS = datetime(1990, 1, 1, tzinfo=timezone.utc)
_MAX_TS = datetime(2100, 1, 1, tzinfo=timezone.utc)


class InputError(ValueError):
    """Raised when the input JSON is malformed or missing required fields."""


def sanitize_display(raw: Any, max_chars: int = MAX_HANDLE_CHARS) -> str:
    """Make an untrusted string safe to print to a terminal.

    Drops control, format (bidi overrides, zero-width), surrogate, private-use
    and line/paragraph-separator characters, then truncates.
    """
    text = str(raw)[: max_chars * 4]
    kept = [
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp")
    ]
    return "".join(kept)[:max_chars]


def _opt_count(raw: Any, where: str) -> Optional[int]:
    """Validate an optional non-negative integer count."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise InputError(f"{where}: must be a non-negative integer, got {type(raw).__name__}")
    if isinstance(raw, float):
        if not math.isfinite(raw) or raw != int(raw):
            raise InputError(f"{where}: must be a non-negative integer")
        raw = int(raw)
    if raw < 0 or raw > MAX_COUNT:
        raise InputError(f"{where}: must be between 0 and {MAX_COUNT}")
    return raw


def _opt_bool(raw: Any, where: str) -> Optional[bool]:
    if raw is None or isinstance(raw, bool):
        return raw
    raise InputError(f"{where}: must be true, false or null, got {type(raw).__name__}")


def _opt_str(raw: Any, where: str) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (str, int)) and not isinstance(raw, bool):
        return str(raw)
    raise InputError(f"{where}: must be a string, got {type(raw).__name__}")


def _reject_constant(name: str) -> Any:
    raise InputError(f"non-finite number {name} is not allowed")


def _parse_timestamp(raw: Any, where: str) -> datetime:
    if not isinstance(raw, str):
        raise InputError(f"{where}: timestamp must be a string, got {type(raw).__name__}")
    text = raw.strip()
    # datetime.fromisoformat only learned to parse a trailing 'Z' in 3.11; we
    # support 3.9+, so normalize it ourselves.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise InputError(f"{where}: could not parse ISO-8601 timestamp {raw!r}: {exc}") from exc
    if dt.tzinfo is None:
        # Assume UTC for naive timestamps rather than crashing; note it nowhere
        # so the caller is responsible for collecting consistent data.
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        dt = dt.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise InputError(f"{where}: timestamp out of range") from exc
    if not (_MIN_TS <= dt < _MAX_TS):
        raise InputError(
            f"{where}: timestamp must be between {_MIN_TS.year} and {_MAX_TS.year}"
        )
    return dt


@dataclass
class Post:
    """A single post (original, reply, or repost) authored by the account."""

    timestamp: datetime
    text: str
    id: Optional[str] = None
    is_reply: Optional[bool] = None
    in_reply_to: Optional[str] = None
    likes: Optional[int] = None
    reposts: Optional[int] = None
    keywords_targeted: list[str] = field(default_factory=list)

    @property
    def engagement(self) -> Optional[int]:
        """Total received engagement, or None if neither field was collected."""
        if self.likes is None and self.reposts is None:
            return None
        return (self.likes or 0) + (self.reposts or 0)

    @classmethod
    def from_dict(cls, raw: dict, where: str) -> "Post":
        if not isinstance(raw, dict):
            raise InputError(f"{where}: each post must be an object")
        if "text" not in raw:
            raise InputError(f"{where}: missing required field 'text'")
        if "timestamp" not in raw:
            raise InputError(f"{where}: missing required field 'timestamp'")
        kw = raw.get("keywords_targeted") or []
        if not isinstance(kw, list):
            raise InputError(f"{where}: 'keywords_targeted' must be a list")
        if not isinstance(raw["text"], str):
            raise InputError(f"{where}: 'text' must be a string")
        if not all(isinstance(k, str) for k in kw):
            raise InputError(f"{where}: 'keywords_targeted' must be a list of strings")
        return cls(
            timestamp=_parse_timestamp(raw["timestamp"], where),
            text=raw["text"],
            id=_opt_str(raw.get("id"), f"{where}.id"),
            is_reply=_opt_bool(raw.get("is_reply"), f"{where}.is_reply"),
            in_reply_to=_opt_str(raw.get("in_reply_to"), f"{where}.in_reply_to"),
            likes=_opt_count(raw.get("likes"), f"{where}.likes"),
            reposts=_opt_count(raw.get("reposts"), f"{where}.reposts"),
            keywords_targeted=list(kw),
        )


@dataclass
class Account:
    """One account and the posts collected from it, sorted oldest-first."""

    handle: str
    posts: list[Post]
    followers: Optional[int] = None
    following: Optional[int] = None
    created_at: Optional[datetime] = None
    collected_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.posts.sort(key=lambda p: p.timestamp)

    @property
    def span(self) -> Optional[tuple[datetime, datetime]]:
        if not self.posts:
            return None
        return (self.posts[0].timestamp, self.posts[-1].timestamp)

    @classmethod
    def from_dict(cls, raw: dict) -> "Account":
        if not isinstance(raw, dict):
            raise InputError("top-level JSON must be an object")
        posts_raw = raw.get("posts")
        if not isinstance(posts_raw, list) or not posts_raw:
            raise InputError("'posts' must be a non-empty list")
        if len(posts_raw) > MAX_POSTS:
            raise InputError(f"'posts' has more than {MAX_POSTS} entries")
        posts = [Post.from_dict(p, f"posts[{i}]") for i, p in enumerate(posts_raw)]
        account_meta = raw.get("account") or {}
        if not isinstance(account_meta, dict):
            raise InputError("'account' must be an object")
        created = account_meta.get("created_at")
        collected = raw.get("collected_at")
        handle_raw = raw.get("handle")
        if handle_raw is not None and not isinstance(handle_raw, str):
            raise InputError("'handle' must be a string")
        return cls(
            handle=sanitize_display(handle_raw or "") or "unknown",
            posts=posts,
            followers=_opt_count(account_meta.get("followers"), "account.followers"),
            following=_opt_count(account_meta.get("following"), "account.following"),
            created_at=_parse_timestamp(created, "account.created_at") if created else None,
            collected_at=_parse_timestamp(collected, "collected_at") if collected else None,
        )


def load_account(path: str) -> Account:
    """Load and validate an account JSON file from ``path``."""
    try:
        with open(path, "rb") as fh:
            # Regular files only: a FIFO or device node could block or never end.
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                raise InputError(f"{path}: not a regular file")
            data = fh.read(MAX_INPUT_BYTES + 1)
    except IsADirectoryError as exc:
        raise InputError(f"{path}: not a regular file") from exc
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise InputError(f"{path}: cannot read file: {exc.strerror or exc}") from exc
    if len(data) > MAX_INPUT_BYTES:
        raise InputError(f"{path}: input larger than {MAX_INPUT_BYTES} bytes")
    try:
        raw = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except InputError as exc:
        raise InputError(f"{path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: not valid UTF-8") from exc
    except RecursionError as exc:
        raise InputError(f"{path}: JSON is nested too deeply") from exc
    except ValueError as exc:  # JSONDecodeError, oversized integer literals
        raise InputError(f"{path}: invalid JSON: {exc}") from exc
    return Account.from_dict(raw)
