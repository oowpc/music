import os
from collections.abc import Iterable

from music21 import chord, converter, note, tempo

from src.models.melody_curve import MelodyCurve
from src.models.note import Note


def _first_tempo_bpm(score) -> float:
    """Return the first explicit tempo in BPM, defaulting to 120."""
    for mark in score.recurse().getElementsByClass(tempo.MetronomeMark):
        if mark.number:
            return float(mark.number)
    return 120.0


def _element_offset(element, score) -> float:
    """Return an element offset in quarter-note units relative to the score."""
    try:
        return float(element.getOffsetInHierarchy(score))
    except Exception:
        return float(element.offset)


def _element_velocity(element) -> int:
    """Return MIDI velocity in the 0-127 range, defaulting to 0 if missing."""
    velocity = getattr(element.volume, "velocity", None)
    if velocity is None:
        return 0
    return int(velocity)


def _extract_notes(score) -> list[Note]:
    """Extract note events from a music21 score as seconds, pitch, and velocity."""
    bpm = _first_tempo_bpm(score)
    seconds_per_quarter = 60.0 / bpm
    notes: list[Note] = []

    for element in score.recurse().notes:
        timestamp = _element_offset(element, score) * seconds_per_quarter
        velocity = _element_velocity(element)

        if isinstance(element, note.Note):
            notes.append(
                Note(
                    timestamp=timestamp,
                    pitch=int(element.pitch.midi),
                    velocity=velocity,
                )
            )
        elif isinstance(element, chord.Chord):
            for pitch in element.pitches:
                notes.append(
                    Note(
                        timestamp=timestamp,
                        pitch=int(pitch.midi),
                        velocity=velocity,
                    )
                )

    notes.sort(key=lambda midi_note: (midi_note.timestamp, midi_note.pitch))
    return notes


def load_midi(filepath: str) -> MelodyCurve | None:
    """Parse a MIDI file into a MelodyCurve, returning None on failure."""
    if not filepath.lower().endswith((".mid", ".midi")):
        return None

    try:
        score = converter.parse(filepath)
    except Exception:
        return None

    raw_notes = _extract_notes(score)
    if not raw_notes:
        return None

    return MelodyCurve(
        name=os.path.splitext(os.path.basename(filepath))[0],
        filepath=filepath,
        raw_notes=raw_notes,
    )


def load_midi_files(filepaths: Iterable[str]) -> list[MelodyCurve]:
    """Load multiple MIDI files, skipping files that cannot be parsed."""
    curves: list[MelodyCurve] = []
    for filepath in filepaths:
        curve = load_midi(filepath)
        if curve is not None:
            curves.append(curve)
    return curves
