"""Behavioral signals: timing, length, ratio, asymmetry, bursts.

These describe *how* an account posts rather than *what* it writes. They tend to
be the cheapest to compute and the cheapest for an informed operator to evade, so
weight them as supporting evidence, not as a verdict.
"""

from __future__ import annotations

from collections import Counter

from ..model import Account
from ..result import Signal
from .. import textutil as tu

_MIN_POSTS = 8  # below this, timing/length distributions are too thin to trust


def reply_timing_consistency(acct: Account) -> Signal:
    key, fam = "reply_timing_consistency", "behavioral"
    posts = acct.posts
    if len(posts) < _MIN_POSTS:
        return Signal(key, fam, None, f"need >= {_MIN_POSTS} posts (have {len(posts)})")
    gaps = [
        (b.timestamp - a.timestamp).total_seconds()
        for a, b in zip(posts, posts[1:])
    ]
    gaps = [g for g in gaps if g > 0]
    if len(gaps) < 3:
        return Signal(key, fam, None, "too few distinct inter-post gaps")
    value = tu.uniformity_score(gaps)
    cv = tu.coefficient_of_variation(gaps)
    note = (
        f"tight inter-post gaps (cv={cv:.2f})"
        if value > 0.6
        else f"human-like timing variance (cv={cv:.2f})"
    )
    return Signal(key, fam, value, note)


def length_uniformity(acct: Account) -> Signal:
    key, fam = "length_uniformity", "behavioral"
    lengths = [len(p.text) for p in acct.posts if p.text.strip()]
    if len(lengths) < _MIN_POSTS:
        return Signal(key, fam, None, f"need >= {_MIN_POSTS} non-empty posts")
    value = tu.uniformity_score(lengths)
    cv = tu.coefficient_of_variation(lengths)
    note = (
        f"lengths cluster narrowly (cv={cv:.2f})"
        if value > 0.6
        else f"varied post lengths (cv={cv:.2f})"
    )
    return Signal(key, fam, value, note)


def reply_ratio(acct: Account) -> Signal:
    key, fam = "reply_ratio", "behavioral"
    flagged = [p for p in acct.posts if p.is_reply is not None]
    if not flagged:
        return Signal(key, fam, None, "no posts carry is_reply")
    replies = sum(1 for p in flagged if p.is_reply)
    value = replies / len(flagged)
    note = f"{value * 100:.0f}% of posts are replies"
    return Signal(key, fam, value, note)


def engagement_asymmetry(acct: Account) -> Signal:
    key, fam = "engagement_asymmetry", "behavioral"
    eng = [p.engagement for p in acct.posts if p.engagement is not None]
    if not eng:
        return Signal(key, fam, None, "no likes/reposts collected")
    avg_eng = tu.mean(eng)
    # Many outputs earning little engagement -> high asymmetry. Map average
    # received engagement per post through a decaying curve: 0 engagement -> 1.0,
    # rising engagement -> toward 0.
    value = tu.clamp01(1.0 / (1.0 + avg_eng))
    note = (
        f"emits >> receives (avg {avg_eng:.1f} eng/post)"
        if value > 0.6
        else f"earns engagement (avg {avg_eng:.1f} eng/post)"
    )
    return Signal(key, fam, value, note)


def burst_pattern(acct: Account) -> Signal:
    key, fam = "burst_pattern", "behavioral"
    posts = acct.posts
    if len(posts) < _MIN_POSTS:
        return Signal(key, fam, None, f"need >= {_MIN_POSTS} posts")
    # Bucket posts into hour-of-history bins and look for high-rate hours.
    start = posts[0].timestamp
    by_hour: Counter[int] = Counter()
    for p in posts:
        hour = int((p.timestamp - start).total_seconds() // 3600)
        by_hour[hour] += 1
    rates = list(by_hour.values())
    if not rates:
        return Signal(key, fam, None, "could not bin posts")
    peak = max(rates)
    threshold = max(5, 3 * (tu.mean(rates)))  # a "burst" is well above baseline
    bursts = [r for r in rates if r >= threshold]
    burst_posts = sum(bursts)
    fraction = burst_posts / len(posts)
    # Combine how concentrated bursts are with how extreme the peak is.
    value = tu.clamp01(0.6 * fraction + 0.4 * tu.clamp01((peak - 5) / 30))
    note = (
        f"{len(bursts)} burst hr(s), peak {peak}/hr"
        if bursts
        else f"no bursts (peak {peak}/hr)"
    )
    return Signal(key, fam, value, note)


SIGNALS = (
    reply_timing_consistency,
    length_uniformity,
    reply_ratio,
    engagement_asymmetry,
    burst_pattern,
)
