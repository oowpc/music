import os
import tempfile

import pytest
from music21 import midi, note, stream, tempo


def _create_midi_file(filepath, notes_data, bpm=120):
    """Write a MIDI file from tuples: (pitch, start_beat, duration_beats, velocity)."""
    score = stream.Stream()
    score.append(tempo.MetronomeMark(number=bpm))

    part = stream.Part()
    for pitch_val, start, duration, velocity in notes_data:
        midi_note = note.Note(pitch_val)
        midi_note.duration.quarterLength = duration
        midi_note.volume.velocity = velocity
        part.insert(start, midi_note)

    score.insert(0, part)
    midi_file = midi.translate.streamToMidiFile(score)
    midi_file.open(filepath, "wb")
    midi_file.write()
    midi_file.close()


@pytest.fixture
def simple_midi_file():
    """A simple 8-note ascending scale MIDI: C4..C5, quarter notes."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "simple_scale.mid")
    notes = [(60 + i, float(i), 1.0, 80) for i in range(8)]
    _create_midi_file(filepath, notes, bpm=120)
    yield filepath
    os.remove(filepath)
    os.rmdir(tmpdir)


@pytest.fixture
def two_note_midi_file():
    """Minimal MIDI with two notes for edge case testing."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "two_notes.mid")
    notes = [(60, 0.0, 1.0, 100), (64, 1.0, 1.0, 90)]
    _create_midi_file(filepath, notes, bpm=120)
    yield filepath
    os.remove(filepath)
    os.rmdir(tmpdir)


@pytest.fixture
def sibling_midi_file():
    """A second MIDI with similar but shifted melody for distance testing."""
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "shifted.mid")
    notes = [(62 + i, float(i), 1.0, 80) for i in range(8)]
    _create_midi_file(filepath, notes, bpm=120)
    yield filepath
    os.remove(filepath)
    os.rmdir(tmpdir)
