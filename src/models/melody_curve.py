from dataclasses import dataclass, field

import numpy as np

from src.models.note import Note


@dataclass
class MelodyCurve:
    """A melody curve with raw MIDI notes and optional normalized 3D points."""

    name: str
    filepath: str
    label: str | None = None
    raw_notes: list[Note] = field(default_factory=list)
    points: np.ndarray | None = None
    color: str = "#ffffff"

    @property
    def num_notes(self) -> int:
        """Return the number of raw notes in the curve."""
        return len(self.raw_notes)
