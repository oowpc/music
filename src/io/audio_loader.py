"""Audio-to-MIDI loader using Spotify Basic Pitch for MP3/WAV → MelodyCurve.

Uses basic-pitch to transcribe audio into ``pretty_midi.PrettyMIDI``, writes a
temporary MIDI file, and loads it through music21 to produce ``MelodyCurve``
objects compatible with the existing Hausdorff / clustering pipeline.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

from src.models.melody_curve import MelodyCurve
from src.models.note import Note

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable transcription parameters
# ---------------------------------------------------------------------------

DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 127.7
DEFAULT_MELODIA_TRICK = True
DEFAULT_MINIMUM_FREQUENCY: float | None = None
DEFAULT_MAXIMUM_FREQUENCY: float | None = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transcribe_to_midi(
    audio_path: str,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    minimum_frequency: float | None = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float | None = DEFAULT_MAXIMUM_FREQUENCY,
    melodia_trick: bool = DEFAULT_MELODIA_TRICK,
) -> str:
    """Transcribe *audio_path* via Basic Pitch and write a temporary MIDI file.

    Returns the path to the temporary ``.mid`` file (caller must delete).
    """
    _, midi_data, _ = predict(
        audio_path,
        ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        melodia_trick=melodia_trick,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    midi_data.write(tmp.name)
    tmp.close()
    return tmp.name


def _extract_notes_from_midi(
    midi_path: str,
    extraction_mode: str = "all",
) -> list[Note]:
    """Parse a MIDI file through music21 and extract note events.

    *extraction_mode* follows the same convention as :func:`src.io.midi_loader.load_midi`:
    ``"all"`` keeps every note, ``"highest"`` picks the top pitch per timestamp,
    ``"strongest"`` picks the loudest.
    """
    if extraction_mode not in {"all", "highest", "strongest"}:
        raise ValueError(f"Unsupported extraction mode: {extraction_mode}")

    from music21 import chord, converter, note as m21_note

    try:
        score = converter.parse(midi_path)
    except Exception:
        return []

    raw: list[Note] = []
    for element in score.flatten().notes:
        offset = float(getattr(element, "offset", 0.0))
        velocity = getattr(getattr(element, "volume", None), "velocity", 80) or 80
        if isinstance(element, m21_note.Note):
            raw.append(Note(timestamp=offset, pitch=int(element.pitch.midi), velocity=velocity))
        elif isinstance(element, chord.Chord):
            for pitch in element.pitches:
                raw.append(Note(timestamp=offset, pitch=int(pitch.midi), velocity=velocity))

    raw.sort(key=lambda n: (n.timestamp, n.pitch))
    return _collapse_simultaneous(raw, extraction_mode)


def _collapse_simultaneous(notes: list[Note], mode: str) -> list[Note]:
    """Reduce simultaneous notes to one per timestamp."""
    if mode == "all":
        return notes

    grouped: dict[float, list[Note]] = {}
    for n in notes:
        grouped.setdefault(round(n.timestamp, 9), []).append(n)

    collapsed: list[Note] = []
    for group in grouped.values():
        if mode == "highest":
            selected = max(group, key=lambda n: (n.pitch, n.velocity))
        else:
            selected = max(group, key=lambda n: (n.velocity, n.pitch))
        collapsed.append(selected)

    collapsed.sort(key=lambda n: (n.timestamp, n.pitch))
    return collapsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_audio(
    filepath: str,
    extraction_mode: str = "highest",
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    minimum_frequency: float | None = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float | None = DEFAULT_MAXIMUM_FREQUENCY,
    melodia_trick: bool = DEFAULT_MELODIA_TRICK,
) -> MelodyCurve | None:
    """Transcribe an audio file into a ``MelodyCurve``.

    Supported formats: WAV, MP3, FLAC, OGG, M4A (anything librosa can read).

    *extraction_mode* controls how polyphonic MIDI output is reduced to a
    monophonic melody curve:

    - ``"all"`` — keep every detected note (creates a dense point cloud)
    - ``"highest"`` — per timestamp, keep only the highest pitch (default)
    - ``"strongest"`` — per timestamp, keep the loudest note

    Threshold parameters are forwarded directly to ``basic_pitch.inference.predict``.
    Lower ``onset_threshold`` catches quieter attacks; higher ``minimum_note_length``
    filters out percussion transients.  See the module-level defaults for
    recommended starting values.

    Returns *None* when transcription produces no usable notes.
    """
    if extraction_mode not in {"all", "highest", "strongest"}:
        raise ValueError(f"Unsupported extraction mode: {extraction_mode}")

    midi_path: str | None = None
    try:
        midi_path = _transcribe_to_midi(
            filepath,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            melodia_trick=melodia_trick,
        )
    except Exception:
        logger.debug("Basic Pitch transcription failed for %s", filepath, exc_info=True)
        return None

    try:
        raw_notes = _extract_notes_from_midi(midi_path, extraction_mode)
    finally:
        try:
            os.unlink(midi_path)
        except OSError:
            pass

    if not raw_notes:
        return None

    name = Path(filepath).stem
    return MelodyCurve(
        name=name,
        filepath=filepath,
        raw_notes=raw_notes,
    )


def load_audio_files(
    filepaths: Iterable[str],
    extraction_mode: str = "highest",
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    minimum_frequency: float | None = DEFAULT_MINIMUM_FREQUENCY,
    maximum_frequency: float | None = DEFAULT_MAXIMUM_FREQUENCY,
    melodia_trick: bool = DEFAULT_MELODIA_TRICK,
) -> list[MelodyCurve]:
    """Load multiple audio files into ``MelodyCurve`` objects, skipping failures.

    Parameters are the same as :func:`load_audio`.
    """
    curves: list[MelodyCurve] = []
    for fp in filepaths:
        curve = load_audio(
            fp,
            extraction_mode=extraction_mode,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            melodia_trick=melodia_trick,
        )
        if curve is not None:
            curves.append(curve)
    return curves
