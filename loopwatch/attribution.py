"""Probabilistic model attribution.

This is the weakest claim loopwatch makes. It compares a handful of coarse style
features against hand-set *priors* for each candidate model family and returns a
probability distribution over "which family does this most resemble." Fingerprints
are statistical priors over short text, not signatures: they shift with every
release, system prompt, sampling temperature, and post-processing step, and any
fine-tune or paraphrase pass blurs them toward noise. Read the README's "Model
attribution" caveats before quoting a number from here.

The priors below are deliberately conservative and easy to inspect/override. They
are starting points for an analyst, not ground truth.
"""

from __future__ import annotations

import math
import re

from .model import Account
from . import textutil as tu

# Feature keys, in a fixed order so profiles line up with extracted vectors.
_FEATURES = (
    "em_dash_per_1k",   # em-dashes per 1000 chars
    "hedge_rate",       # share of posts containing a hedge marker
    "list_rate",        # share of posts using a bullet/numbered list marker
    "avg_sent_len",     # mean sentence length in tokens (scaled /30)
    "exclaim_rate",     # exclamation marks per post (scaled /3)
    "question_rate",    # question marks per post (scaled /3)
    "emoji_rate",       # emoji per post (scaled /3)
)

# Per-family priors over the (scaled) feature space. Values are rough and meant to
# be overridden with locally-derived fingerprints — see CONTRIBUTING.
_PROFILES: dict[str, dict[str, float]] = {
    "Hermes 3":          dict(em_dash_per_1k=1.2, hedge_rate=0.10, list_rate=0.20, avg_sent_len=0.55, exclaim_rate=0.10, question_rate=0.15, emoji_rate=0.05),
    "Hermes Agent":      dict(em_dash_per_1k=1.0, hedge_rate=0.08, list_rate=0.45, avg_sent_len=0.50, exclaim_rate=0.08, question_rate=0.12, emoji_rate=0.03),
    "GPT-4o mini":       dict(em_dash_per_1k=2.0, hedge_rate=0.18, list_rate=0.35, avg_sent_len=0.60, exclaim_rate=0.20, question_rate=0.20, emoji_rate=0.25),
    "GPT-5 series":      dict(em_dash_per_1k=2.4, hedge_rate=0.12, list_rate=0.40, avg_sent_len=0.65, exclaim_rate=0.12, question_rate=0.18, emoji_rate=0.15),
    "Mistral":           dict(em_dash_per_1k=0.7, hedge_rate=0.14, list_rate=0.25, avg_sent_len=0.50, exclaim_rate=0.10, question_rate=0.15, emoji_rate=0.05),
    "Claude":            dict(em_dash_per_1k=2.8, hedge_rate=0.30, list_rate=0.30, avg_sent_len=0.70, exclaim_rate=0.06, question_rate=0.16, emoji_rate=0.02),
    "DeepSeek V4":       dict(em_dash_per_1k=1.0, hedge_rate=0.16, list_rate=0.50, avg_sent_len=0.62, exclaim_rate=0.08, question_rate=0.14, emoji_rate=0.04),
    "OpenClaw":          dict(em_dash_per_1k=1.5, hedge_rate=0.10, list_rate=0.30, avg_sent_len=0.55, exclaim_rate=0.15, question_rate=0.20, emoji_rate=0.10),
    "raw Llama":         dict(em_dash_per_1k=0.5, hedge_rate=0.20, list_rate=0.15, avg_sent_len=0.45, exclaim_rate=0.12, question_rate=0.18, emoji_rate=0.08),
}

# Feature weights: down-weight the noisier tells (em-dash, emoji) so no single
# fragile feature dominates the distance.
_WEIGHTS = dict(
    em_dash_per_1k=0.6, hedge_rate=1.0, list_rate=1.0, avg_sent_len=1.0,
    exclaim_rate=0.7, question_rate=0.7, emoji_rate=0.5,
)

# A broad emoji range; good enough for a rate feature without a dependency.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_LIST_RE = re.compile(r"(^|\n)\s*([-*•]|\d+[.)])\s+")


def extract_features(acct: Account) -> dict[str, float]:
    posts = [p for p in acct.posts if p.text.strip()]
    n = len(posts)
    total_chars = sum(len(p.text) for p in posts) or 1
    em = sum(p.text.count(tu.EM_DASH) for p in posts)
    hedged = sum(
        1 for p in posts if any(m in p.text.lower() for m in tu.HEDGE_MARKERS)
    )
    listed = sum(1 for p in posts if _LIST_RE.search(p.text))
    exclaim = sum(p.text.count("!") for p in posts)
    question = sum(p.text.count("?") for p in posts)
    emoji = sum(len(_EMOJI_RE.findall(p.text)) for p in posts)
    sent_lens = [
        len(tu.tokenize(s)) for p in posts for s in tu.sentences(p.text)
    ]
    avg_sent = tu.mean(sent_lens) if sent_lens else 0.0
    n = n or 1
    return {
        "em_dash_per_1k": em / (total_chars / 1000.0),
        "hedge_rate": hedged / n,
        "list_rate": listed / n,
        "avg_sent_len": tu.clamp01(avg_sent / 30.0),
        "exclaim_rate": tu.clamp01((exclaim / n) / 3.0),
        "question_rate": tu.clamp01((question / n) / 3.0),
        "emoji_rate": tu.clamp01((emoji / n) / 3.0),
    }


def _distance(feats: dict[str, float], profile: dict[str, float]) -> float:
    total = 0.0
    for k in _FEATURES:
        w = _WEIGHTS[k]
        total += w * (feats[k] - profile[k]) ** 2
    return math.sqrt(total)


def attribute(acct: Account) -> dict:
    """Return a probabilistic attribution result.

    Result shape::

        {
          "distribution": {model: probability, ...},   # sums to 1.0
          "top": model,
          "confidence": float,    # spread between #1 and #2; low == unreliable
          "features": {...},
          "reliable": bool,       # False when there's too little text to trust
        }
    """
    feats = extract_features(acct)
    total_chars = sum(len(p.text) for p in acct.posts)
    reliable = len(acct.posts) >= 30 and total_chars >= 2000

    sims = {}
    for name, profile in _PROFILES.items():
        d = _distance(feats, profile)
        sims[name] = math.exp(-d)  # closer profile -> higher similarity
    z = sum(sims.values()) or 1.0
    dist = {name: s / z for name, s in sims.items()}
    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    top, top_p = ranked[0]
    second_p = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = top_p - second_p
    return {
        "distribution": dist,
        "ranked": ranked,
        "top": top,
        "confidence": confidence,
        "features": feats,
        "reliable": reliable,
    }
