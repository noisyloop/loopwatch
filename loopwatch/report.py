"""Human-readable rendering of score and attribution results."""

from __future__ import annotations

from .scoring import ScoreResult
from .attribution import attribute as _attribute
from .model import Account

_FAMILY_TITLES = {
    "behavioral": "BEHAVIORAL",
    "stylometric": "STYLOMETRIC",
    "temporal": "TEMPORAL / GROWTH",
}
_FAMILY_ORDER = ("behavioral", "stylometric", "temporal")


def _fmt_value(v) -> str:
    return f"{v:.2f}" if v is not None else "  -- "


def render_score(result: ScoreResult, acct: Account, top_n_models: int = 4) -> str:
    lines: list[str] = []
    span = result.span
    span_str = (
        f"{span[0].date()} .. {span[1].date()}" if span else "n/a"
    )
    lines.append(
        f"account: @{result.handle}    posts: {result.n_posts}    span: {span_str}"
    )
    lines.append("")

    for fam in _FAMILY_ORDER:
        fam_sigs = [s for s in result.signals if s.family == fam]
        if not fam_sigs:
            continue
        lines.append(f"  {_FAMILY_TITLES[fam]}")
        width = max(len(s.key) for s in fam_sigs)
        for s in fam_sigs:
            lines.append(
                f"    {s.key.ljust(width)}   {_fmt_value(s.value)}   {s.note}"
            )
        lines.append("")

    # Attribution block.
    attr = _attribute(acct)
    lines.append("  ATTRIBUTION (probabilistic — see limitations)")
    shown = 0
    for name, p in attr["ranked"]:
        if shown >= top_n_models or p < 0.10:
            break
        lines.append(f"    {name.ljust(18)} {p:.2f}")
        shown += 1
    remaining = len(attr["ranked"]) - shown
    if remaining > 0:
        lines.append(f"    (others < 0.10)")
    if not attr["reliable"]:
        lines.append("    note: low text volume — attribution near prior, unreliable")
    lines.append("")

    if result.combined is None:
        lines.append(
            f"  COMBINED SUSPICION   --   /  1.00     [{result.band}]"
        )
        lines.append(
            f"  ({result.n_computed} signals computed of {len(result.signals)})"
        )
    else:
        lines.append(
            f"  COMBINED SUSPICION   {result.combined:.2f}  /  1.00     [{result.band}]"
        )
        lines.append(
            f"  ({result.n_computed} of {len(result.signals)} signals computed, "
            f"window {result.window_days}d)"
        )
    return "\n".join(lines)


def render_attribution(acct: Account) -> str:
    attr = _attribute(acct)
    lines = [f"account: @{acct.handle}    posts: {len(acct.posts)}", ""]
    lines.append("  MODEL ATTRIBUTION (probabilistic)")
    for name, p in attr["ranked"]:
        bar = "#" * int(round(p * 30))
        lines.append(f"    {name.ljust(18)} {p:.3f}  {bar}")
    lines.append("")
    lines.append(f"  top: {attr['top']}   confidence (top1-top2): {attr['confidence']:.3f}")
    if not attr["reliable"]:
        lines.append("  note: low text volume — distribution is near the prior, treat as unreliable")
    lines.append("")
    lines.append("  features")
    width = max(len(k) for k in attr["features"])
    for k, v in attr["features"].items():
        lines.append(f"    {k.ljust(width)}  {v:.3f}")
    return "\n".join(lines)
