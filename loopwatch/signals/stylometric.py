"""Stylometric signals: coherence, vocabulary decay, hedging, em-dashes, entropy.

These describe *what* the account writes and how it writes it. Stylometry is a
moving target (see the README "stylometric arms race" section); treat every value
here as having a shelf life and re-validate against your own labeled data.
"""

from __future__ import annotations

import random
from collections import Counter

from ..model import Account
from ..result import Signal
from .. import textutil as tu

# Human em-dash baseline: em-dashes per 1000 characters. A rough prior; the
# baseline is drifting upward as LLM text saturates the web, so this signal is
# weighted low in scoring.
_EM_DASH_BASELINE_PER_1K = 0.6
_MIN_POSTS = 8


def topic_coherence(acct: Account) -> Signal:
    key, fam = "topic_coherence", "stylometric"
    token_sets = [set(tu.content_tokens(p.text)) for p in acct.posts]
    token_sets = [s for s in token_sets if s]
    if len(token_sets) < _MIN_POSTS:
        return Signal(key, fam, None, f"need >= {_MIN_POSTS} contentful posts")
    # Average pairwise Jaccard over a bounded random sample of post pairs.
    rng = random.Random(1729)  # fixed seed -> deterministic, reproducible runs
    pairs = []
    n = len(token_sets)
    sample = min(400, n * (n - 1) // 2)
    for _ in range(sample):
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            pairs.append(tu.jaccard(token_sets[i], token_sets[j]))
    if not pairs:
        return Signal(key, fam, None, "could not sample post pairs")
    avg = tu.mean(pairs)
    # Human accounts roam across topics; replies in particular share few content
    # words. Sustained overlap reads as narrow, prompted output. Scale modest
    # Jaccard values up since cross-post overlap is naturally small.
    value = tu.clamp01(avg * 4.0)
    note = (
        f"unusually on-topic (avg overlap {avg:.2f})"
        if value > 0.6
        else f"topically varied (avg overlap {avg:.2f})"
    )
    return Signal(key, fam, value, note)


def vocabulary_decay(acct: Account) -> Signal:
    key, fam = "vocabulary_decay", "stylometric"
    posts = [p for p in acct.posts if tu.content_tokens(p.text)]
    if len(posts) < max(_MIN_POSTS, 12):
        return Signal(key, fam, None, "need >= 12 contentful posts")
    # Per-post novelty: fraction of a post's content tokens never seen before.
    seen: set[str] = set()
    novelty: list[float] = []
    for p in posts:
        toks = tu.content_tokens(p.text)
        new = sum(1 for t in toks if t not in seen)
        novelty.append(new / len(toks))
        seen.update(toks)
    # Novelty always trends down (the seen-set only grows); we care about the
    # *back half* collapsing relative to the front half.
    half = len(novelty) // 2
    front, back = tu.mean(novelty[:half]), tu.mean(novelty[half:])
    if front == 0:
        return Signal(key, fam, None, "degenerate novelty profile")
    drop = tu.clamp01((front - back) / front)
    value = drop
    note = f"new-token rate down {drop * 100:.0f}% (late vs early)"
    return Signal(key, fam, value, note)


def hedge_absence(acct: Account) -> Signal:
    key, fam = "hedge_absence", "stylometric"
    texts = [p.text.lower() for p in acct.posts if p.text.strip()]
    if len(texts) < _MIN_POSTS:
        return Signal(key, fam, None, f"need >= {_MIN_POSTS} non-empty posts")
    hedged = 0
    for t in texts:
        if any(marker in t for marker in tu.HEDGE_MARKERS):
            hedged += 1
    hedge_rate = hedged / len(texts)
    # Absence of hedging -> high score. A confidently un-hedged stream of replies
    # is a weak generation tell.
    value = tu.clamp01(1.0 - hedge_rate * 3.0)
    note = f"hedging in {hedge_rate * 100:.0f}% of posts"
    return Signal(key, fam, value, note)


def em_dash_frequency(acct: Account) -> Signal:
    key, fam = "em_dash_frequency", "stylometric"
    total_chars = sum(len(p.text) for p in acct.posts)
    if total_chars < 500:
        return Signal(key, fam, None, "too little text for a stable rate")
    em = sum(p.text.count(tu.EM_DASH) for p in acct.posts)
    per_1k = em / (total_chars / 1000.0)
    ratio = per_1k / _EM_DASH_BASELINE_PER_1K if _EM_DASH_BASELINE_PER_1K else 0.0
    # Map the multiple-of-baseline through a gentle curve; 1x baseline -> ~0.3,
    # 3x -> ~0.7. This is intentionally a weak tell (see README).
    value = tu.clamp01((ratio - 0.5) / 3.5)
    note = f"em-dashes at {ratio:.1f}x human baseline ({per_1k:.2f}/1k chars)"
    return Signal(key, fam, value, note)


def sentence_entropy(acct: Account) -> Signal:
    key, fam = "sentence_entropy", "stylometric"
    lengths: list[int] = []
    for p in acct.posts:
        for s in tu.sentences(p.text):
            lengths.append(len(tu.tokenize(s)))
    lengths = [n for n in lengths if n > 0]
    if len(lengths) < 12:
        return Signal(key, fam, None, "need >= 12 sentences")
    # Entropy of the sentence-length distribution (binned). Low structural
    # variation -> low entropy -> high (suspicious) score.
    bins = Counter(min(n, 40) // 4 for n in lengths)
    h = tu.shannon_entropy(bins.values())
    # Normalize against the maximum possible entropy for the number of bins used.
    max_h = tu.shannon_entropy([1] * len(bins)) or 1.0
    norm = h / max_h
    value = tu.clamp01(1.0 - norm)
    note = f"low sentence-structure variation (norm entropy {norm:.2f})"
    return Signal(key, fam, value, note)


SIGNALS = (
    topic_coherence,
    vocabulary_decay,
    hedge_absence,
    em_dash_frequency,
    sentence_entropy,
)
