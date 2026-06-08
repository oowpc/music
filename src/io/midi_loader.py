import os
from collections.abc import Iterable
from dataclasses import dataclass

from music21 import chord, converter, instrument, note, tempo

from src.models.melody_curve import MelodyCurve
from src.models.note import Note


@dataclass(frozen=True)
class MidiTrackInfo:
    """Summary metadata for one MIDI part/track."""

    index: int
    name: str
    instrument_name: str
    note_count: int
    pitch_min: int | None
    pitch_max: int | None
    start_seconds: float | None
    end_seconds: float | None


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


def _track_sources(score) -> list:
    """Return score parts when present, otherwise a single whole-score source."""
    parts = list(score.parts)
    return parts if parts else [score]


def _instrument_name(source) -> str:
    """Return the first instrument name in a stream source, or Unknown."""
    for item in source.recurse().getElementsByClass(instrument.Instrument):
        if item.instrumentName:
            return str(item.instrumentName)
        if item.bestName():
            return str(item.bestName())
    return "Unknown"


def _source_name(source, index: int) -> str:
    """Return a readable name for a MIDI source/part."""
    if getattr(source, "partName", None):
        return str(source.partName)
    if getattr(source, "id", None):
        return str(source.id)
    return f"Track {index}"


def _collapse_simultaneous_notes(notes: list[Note], mode: str) -> list[Note]:
    """Collapse simultaneous notes with highest-pitch or strongest-velocity rules."""
    if mode == "all":
        return notes
    if mode not in {"highest", "strongest"}:
        raise ValueError(f"Unsupported extraction mode: {mode}")

    grouped: dict[float, list[Note]] = {}
    for midi_note in notes:
        grouped.setdefault(round(midi_note.timestamp, 9), []).append(midi_note)

    collapsed = []
    for group in grouped.values():
        if mode == "highest":
            selected = max(group, key=lambda midi_note: (midi_note.pitch, midi_note.velocity))
        else:
            selected = max(group, key=lambda midi_note: (midi_note.velocity, midi_note.pitch))
        collapsed.append(selected)

    collapsed.sort(key=lambda midi_note: (midi_note.timestamp, midi_note.pitch))
    return collapsed


def _extract_notes(source, bpm: float, extraction_mode: str = "all") -> list[Note]:
    """Extract note events from a stream as seconds, pitch, and velocity.

    ``source`` may be a full score or a selected part. The returned notes are
    sorted and have shape-equivalent tuples of (timestamp, pitch, velocity).
    ``extraction_mode`` can be ``all``, ``highest``, or ``strongest``.
    """
    seconds_per_quarter = 60.0 / bpm
    notes: list[Note] = []

    for element in source.recurse().notes:
        timestamp = _element_offset(element, source) * seconds_per_quarter
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
    return _collapse_simultaneous_notes(notes, extraction_mode)


def inspect_midi_tracks(filepath: str) -> list[MidiTrackInfo]:
    """Return track summaries for a MIDI file.

    Invalid files return an empty list. Each summary includes note count, pitch
    range, and approximate start/end time in seconds for one music21 part.
    """
    if not filepath.lower().endswith((".mid", ".midi")):
        return []

    try:
        score = converter.parse(filepath)
    except Exception:
        return []

    bpm = _first_tempo_bpm(score)
    infos = []
    for index, source in enumerate(_track_sources(score)):
        notes = _extract_notes(source, bpm, extraction_mode="all")
        pitches = [midi_note.pitch for midi_note in notes]
        timestamps = [midi_note.timestamp for midi_note in notes]
        infos.append(
            MidiTrackInfo(
                index=index,
                name=_source_name(source, index),
                instrument_name=_instrument_name(source),
                note_count=len(notes),
                pitch_min=min(pitches) if pitches else None,
                pitch_max=max(pitches) if pitches else None,
                start_seconds=min(timestamps) if timestamps else None,
                end_seconds=max(timestamps) if timestamps else None,
            )
        )
    return infos


def load_midi(
    filepath: str,
    track_index: int | None = None,
    extraction_mode: str = "all",
) -> MelodyCurve | None:
    """Parse a MIDI file into a MelodyCurve, returning None on failure.

    ``track_index`` selects a specific music21 part when provided. With
    ``extraction_mode='highest'`` or ``'strongest'``, simultaneous notes are
    reduced to one melody point per timestamp.
    """
    if extraction_mode not in {"all", "highest", "strongest"}:
        raise ValueError(f"Unsupported extraction mode: {extraction_mode}")
    if not filepath.lower().endswith((".mid", ".midi")):
        return None

    try:
        score = converter.parse(filepath)
    except Exception:
        return None

    sources = _track_sources(score)
    if track_index is not None:
        if track_index < 0 or track_index >= len(sources):
            return None
        source = sources[track_index]
    else:
        source = score

    raw_notes = _extract_notes(source, _first_tempo_bpm(score), extraction_mode)
    if not raw_notes:
        return None

    name = os.path.splitext(os.path.basename(filepath))[0]
    if track_index is not None:
        name = f"{name} - {_source_name(source, track_index)}"
    if extraction_mode != "all":
        name = f"{name} ({extraction_mode})"

    return MelodyCurve(
        name=name,
        filepath=filepath,
        raw_notes=raw_notes,
    )


def load_midi_files(
    filepaths: Iterable[str],
    extraction_mode: str = "all",
) -> list[MelodyCurve]:
    """Load multiple MIDI files, skipping files that cannot be parsed."""
    curves: list[MelodyCurve] = []
    for filepath in filepaths:
        curve = load_midi(filepath, extraction_mode=extraction_mode)
        if curve is not None:
            curves.append(curve)
    return curves
