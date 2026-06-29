"""Temporal / growth signals: the feedback-loop detectors.

These are the signals loopwatch is named for. They look for *change over time* —
an account that tightens its cadence or narrows its vocabulary once it starts
getting attention. Each operates over rolling windows so it keys on inflection,
not absolute level (which is what makes Goodhart-style flooding harder, though not
impossible — see the README).

Temporal signals accept a ``window_days`` argument; the aggregator passes the value
from ``--window``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from ..model import Account
from ..result import Signal
from .. import textutil as tu

_MIN_POSTS = 12


def _bucket_by_day(acct: Account) -> dict[int, list]:
    start = acct.posts[0].timestamp
    buckets: dict[int, list] = defaultdict(list)
    for p in acct.posts:
        day = int((p.timestamp - start).total_seconds() // 86400)
        buckets[day].append(p)
    return buckets


def volume_spike(acct: Account, window_days: int = 7) -> Signal:
    key, fam = "volume_spike", "temporal"
    if len(acct.posts) < _MIN_POSTS:
        return Signal(key, fam, None, f"need >= {_MIN_POSTS} posts")
    buckets = _bucket_by_day(acct)
    span = acct.span
    if not span:
        return Signal(key, fam, None, "no time span")
    last_day = int((span[1] - span[0]).total_seconds() // 86400)
    if last_day < window_days:
        return Signal(key, fam, None, f"history shorter than window ({window_days}d)")
    daily = [len(buckets.get(d, [])) for d in range(last_day + 1)]
    baseline = tu.mean(daily) or 1.0
    peak = max(daily)
    multiple = peak / baseline
    # Count days that exceed 3x baseline as spikes.
    spikes = sum(1 for c in daily if c >= 3 * baseline and c >= 3)
    value = tu.clamp01(0.5 * tu.clamp01((multiple - 2) / 6) + 0.5 * tu.clamp01(spikes / 3))
    note = (
        f"{spikes} spike day(s), peak {multiple:.1f}x baseline"
        if spikes
        else f"flat volume (peak {multiple:.1f}x baseline)"
    )
    return Signal(key, fam, value, note)


def lexical_novelty_decay(acct: Account, window_days: int = 7) -> Signal:
    key, fam = "lexical_novelty_decay", "temporal"
    posts = [p for p in acct.posts if tu.content_tokens(p.text)]
    if len(posts) < _MIN_POSTS:
        return Signal(key, fam, None, f"need >= {_MIN_POSTS} contentful posts")
    buckets = _bucket_by_day(acct)
    span = acct.span
    last_day = int((span[1] - span[0]).total_seconds() // 86400)
    daily = [len(buckets.get(d, [])) for d in range(last_day + 1)]
    if not daily:
        return Signal(key, fam, None, "no daily volume")
    # Anchor on the largest spike day; compare novelty before vs after it.
    spike_day = max(range(len(daily)), key=lambda d: daily[d])
    start = acct.posts[0].timestamp
    cutoff = start + timedelta(days=spike_day)
    before = [p for p in posts if p.timestamp <= cutoff]
    after = [p for p in posts if p.timestamp > cutoff]
    if len(before) < 4 or len(after) < 4:
        return Signal(key, fam, None, "spike too close to an endpoint")

    def novelty_rate(seq, prior_seen):
        seen = set(prior_seen)
        rates = []
        for p in seq:
            toks = tu.content_tokens(p.text)
            new = sum(1 for t in toks if t not in seen)
            rates.append(new / len(toks))
            seen.update(toks)
        return tu.mean(rates)

    before_rate = novelty_rate(before, set())
    after_rate = novelty_rate(after, {t for p in before for t in tu.content_tokens(p.text)})
    if before_rate == 0:
        return Signal(key, fam, None, "degenerate novelty profile")
    drop = tu.clamp01((before_rate - after_rate) / before_rate)
    value = drop
    note = f"novelty down {drop * 100:.0f}% after spike (day {spike_day})"
    return Signal(key, fam, value, note)


def behavioral_drift(acct: Account, window_days: int = 7) -> Signal:
    key, fam = "behavioral_drift", "temporal"
    posts = acct.posts
    if len(posts) < max(_MIN_POSTS, 16):
        return Signal(key, fam, None, "need >= 16 posts for rolling windows")
    span = acct.span
    total_days = (span[1] - span[0]).total_seconds() / 86400
    if total_days < 2 * window_days:
        return Signal(key, fam, None, f"history shorter than 2 windows ({window_days}d)")
    # Slice into consecutive windows; track inter-post gap variability per window.
    start = span[0]
    win = timedelta(days=window_days)
    windows: list[list] = []
    cur: list = []
    edge = start + win
    for p in posts:
        if p.timestamp <= edge:
            cur.append(p)
        else:
            windows.append(cur)
            cur = [p]
            while p.timestamp > edge:
                edge += win
    windows.append(cur)
    windows = [w for w in windows if len(w) >= 3]
    if len(windows) < 2:
        return Signal(key, fam, None, "too few populated windows")

    def window_cv(w):
        gaps = [
            (b.timestamp - a.timestamp).total_seconds()
            for a, b in zip(w, w[1:])
        ]
        gaps = [g for g in gaps if g > 0]
        return tu.coefficient_of_variation(gaps) if len(gaps) >= 2 else None

    cvs = [c for c in (window_cv(w) for w in windows) if c is not None]
    if len(cvs) < 2:
        return Signal(key, fam, None, "insufficient gap data across windows")
    # A *negative* slope (cadence variability falling over time) is the
    # feedback-loop signature: the account is tightening up.
    s = tu.slope(cvs)
    tightening = tu.clamp01(-s / (tu.mean(cvs) or 1.0))
    value = tightening
    note = (
        f"cadence tightening across {len(cvs)} windows"
        if value > 0.5
        else f"stable/loosening cadence across {len(cvs)} windows"
    )
    return Signal(key, fam, value, note)


SIGNALS = (
    volume_spike,
    lexical_novelty_decay,
    behavioral_drift,
)
