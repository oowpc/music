from dataclasses import dataclass


@dataclass
class Note:
    """A MIDI note event represented as time, pitch, and velocity."""

    timestamp: float
    pitch: int
    velocity: int

    def as_tuple(self) -> tuple[float, int, int]:
        """Return the note as ``(timestamp, pitch, velocity)``."""
        return (self.timestamp, self.pitch, self.velocity)
