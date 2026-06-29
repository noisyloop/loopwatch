"""Run all signals over an account and aggregate into a combined suspicion score."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional

from .model import Account
from .result import Signal
from . import signals as _signals

# Per-signal weights in the combined score. Fragile or easily-evaded signals
# (em-dash, length uniformity) are deliberately down-weighted; the feedback-loop
# temporal signals are weighted up because they're the point of the tool. These
# are heuristics, not learned coefficients — tune against your own labeled data.
WEIGHTS: dict[str, float] = {
    # behavioral
    "reply_timing_consistency": 1.0,
    "length_uniformity": 0.7,
    "reply_ratio": 0.8,
    "engagement_asymmetry": 0.9,
    "burst_pattern": 0.9,
    # stylometric
    "topic_coherence": 0.9,
    "vocabulary_decay": 1.0,
    "hedge_absence": 0.8,
    "em_dash_frequency": 0.4,
    "sentence_entropy": 0.7,
    # temporal / growth (the feedback loop)
    "volume_spike": 1.1,
    "lexical_novelty_decay": 1.2,
    "behavioral_drift": 1.3,
}

DEFAULT_WINDOW_DAYS = 7


@dataclass
class ScoreResult:
    handle: str
    signals: list[Signal]
    combined: Optional[float]
    band: str
    n_posts: int
    n_computed: int
    window_days: int
    span: Optional[tuple] = field(default=None)

    def as_dict(self) -> dict:
        return {
            "handle": self.handle,
            "combined": self.combined,
            "band": self.band,
            "n_posts": self.n_posts,
            "n_computed": self.n_computed,
            "window_days": self.window_days,
            "span": [s.isoformat() for s in self.span] if self.span else None,
            "signals": [s.as_dict() for s in self.signals],
        }


def _band(score: Optional[float]) -> str:
    if score is None:
        return "INSUFFICIENT DATA"
    if score >= 0.66:
        return "FLAG"
    if score >= 0.45:
        return "REVIEW"
    return "LIKELY HUMAN"


def _call_signal(fn, acct: Account, window_days: int) -> Signal:
    # Temporal signals accept window_days; the others don't. Inspect once and
    # pass the argument only where it's wanted.
    params = inspect.signature(fn).parameters
    if "window_days" in params:
        return fn(acct, window_days=window_days)
    return fn(acct)


def score_account(acct: Account, window_days: int = DEFAULT_WINDOW_DAYS) -> ScoreResult:
    """Compute every signal and the weighted combined suspicion score."""
    results = [_call_signal(fn, acct, window_days) for fn in _signals.ALL_SIGNALS]

    num = 0.0
    den = 0.0
    for s in results:
        if s.value is None:
            continue
        w = WEIGHTS.get(s.key, 1.0)
        num += w * s.value
        den += w
    combined = (num / den) if den > 0 else None

    return ScoreResult(
        handle=acct.handle,
        signals=results,
        combined=combined,
        band=_band(combined),
        n_posts=len(acct.posts),
        n_computed=sum(1 for s in results if s.computed),
        window_days=window_days,
        span=acct.span,
    )
