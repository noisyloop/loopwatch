"""Input model and JSON loading.

The on-disk format is a single JSON object describing one account and its posts.
Only ``posts[].timestamp`` and ``posts[].text`` are strictly required; every other
field is optional and missing data degrades a signal to ``None`` rather than being
guessed. See the README "Input format" section for the documented schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


class InputError(ValueError):
    """Raised when the input JSON is malformed or missing required fields."""


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
    return dt.astimezone(timezone.utc)


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
        return cls(
            timestamp=_parse_timestamp(raw["timestamp"], where),
            text=str(raw["text"]),
            id=raw.get("id"),
            is_reply=raw.get("is_reply"),
            in_reply_to=raw.get("in_reply_to"),
            likes=raw.get("likes"),
            reposts=raw.get("reposts"),
            keywords_targeted=[str(k) for k in kw],
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
        posts = [Post.from_dict(p, f"posts[{i}]") for i, p in enumerate(posts_raw)]
        account_meta = raw.get("account") or {}
        created = account_meta.get("created_at")
        collected = raw.get("collected_at")
        return cls(
            handle=str(raw.get("handle") or "unknown"),
            posts=posts,
            followers=account_meta.get("followers"),
            following=account_meta.get("following"),
            created_at=_parse_timestamp(created, "account.created_at") if created else None,
            collected_at=_parse_timestamp(collected, "collected_at") if collected else None,
        )


def load_account(path: str) -> Account:
    """Load and validate an account JSON file from ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}: invalid JSON: {exc}") from exc
    return Account.from_dict(raw)
