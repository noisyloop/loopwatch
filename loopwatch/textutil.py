"""Small text and statistics helpers shared by the signal modules.

Deliberately lightweight: regex tokenization and hand-rolled stats so the package
stays pure-stdlib and the math behind every signal is inspectable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")

# A compact stopword list — enough to keep topic/coherence measures focused on
# content words without pulling in a dependency.
STOPWORDS = frozenset(
    """
    a an the and or but if then else for to of in on at by with without from into
    out up down over under again is are was were be been being am do does did doing
    have has had having i you he she it we they me him her us them my your his its our
    their this that these those as so than too very can will just not no nor only own
    same such here there what which who whom when where why how all any both each few
    more most other some about against between through during before after above below
    """.split()
)

# Hedging / uncertainty markers. Their *absence* is a weak LLM-engagement tell.
HEDGE_MARKERS = (
    "maybe", "perhaps", "possibly", "probably", "might", "may", "could",
    "i think", "i guess", "i suppose", "i believe", "seems", "seem", "kind of",
    "sort of", "arguably", "presumably", "apparently", "i'm not sure",
    "not entirely sure", "could be wrong", "in my opinion", "imo", "afaik",
)

EM_DASH = "—"  # —
EN_DASH = "–"  # –


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Word tokens with stopwords removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    """Naive sentence split; good enough for length-distribution stats."""
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: Sequence[float]) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def coefficient_of_variation(xs: Sequence[float]) -> float:
    """Stdev / mean. 0 when perfectly uniform; grows with relative spread."""
    mu = mean(xs)
    if mu == 0:
        return 0.0
    return pstdev(xs) / mu


def shannon_entropy(counts: Iterable[float]) -> float:
    """Shannon entropy (bits) of a count distribution."""
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def uniformity_score(xs: Sequence[float]) -> float:
    """Map a sample's relative spread to a [0,1] *uniformity* score.

    Perfectly uniform values (cv=0) score 1.0; the score decays as the
    coefficient of variation grows. ``exp(-cv)`` gives a smooth, bounded mapping
    with no magic thresholds.
    """
    cv = coefficient_of_variation(xs)
    return clamp01(math.exp(-cv))


def slope(ys: Sequence[float]) -> float:
    """Least-squares slope of ``ys`` against its index (x = 0, 1, 2, ...)."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = mean(xs), mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def top_token_concentration(tokens: Sequence[str], top_n: int = 20) -> float:
    """Share of tokens accounted for by the ``top_n`` most frequent types."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    top = sum(c for _, c in counts.most_common(top_n))
    return top / len(tokens)
