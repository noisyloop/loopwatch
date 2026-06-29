"""Signal implementations, grouped by family.

Each signal is a callable ``(Account) -> Signal`` that returns a normalized value
in ``[0, 1]`` (or ``None`` when the input lacks the data to compute it) plus a short
human-readable note. Signals never raise on missing optional data; they report
``None`` so the aggregator can drop them cleanly.
"""

from . import behavioral, stylometric, temporal

# Ordered the way they appear in the README signal table / sample output.
BEHAVIORAL = behavioral.SIGNALS
STYLOMETRIC = stylometric.SIGNALS
TEMPORAL = temporal.SIGNALS

ALL_SIGNALS = BEHAVIORAL + STYLOMETRIC + TEMPORAL

__all__ = ["BEHAVIORAL", "STYLOMETRIC", "TEMPORAL", "ALL_SIGNALS"]
