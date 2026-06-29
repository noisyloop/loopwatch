"""The ``Signal`` result type, kept in its own module to avoid import cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Signal:
    """The output of one signal.

    ``value`` is normalized to ``[0, 1]`` where higher is more consistent with an
    LLM-in-the-loop account, or ``None`` when the input lacked the data to compute
    it. ``note`` is a short human-readable explanation shown in the report.
    """

    key: str
    family: str
    value: Optional[float]
    note: str

    @property
    def computed(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "family": self.family,
            "value": self.value,
            "note": self.note,
        }
