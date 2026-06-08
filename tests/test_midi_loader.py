import pytest
from music21 import chord, instrument, midi, note, stream, tempo

from src.io.midi_loader import (
    _collapse_simultaneous_notes,
    inspect_midi_tracks,
    load_midi,
    load_midi_files,
)
from src.models.note import Note


def _write_stream(filepath, score):
    """Write a music21 score/stream to a MIDI file."""
    midi_file = midi.translate.streamToMidiFile(score)
    midi_file.open(filepath, "wb")
    midi_file.write()
    midi_file.close()


def _create_multi_track_midi(filepath):
    """Create a two-part MIDI fixture with distinct pitch ranges."""
    score = stream.Score()
    score.append(tempo.MetronomeMark(number=120))

    melody = stream.Part(id="melody")
    melody.partName = "Melody"
    melody.insert(0, instrument.Piano())
    high_note = note.Note(72)
    high_note.volume.velocity = 90
    melody.insert(0.0, high_note)
    next_high_note = note.Note(74)
    next_high_note.volume.velocity = 92
    melody.insert(1.0, next_high_note)

    bass = stream.Part(id="bass")
    bass.partName = "Bass"
    bass.insert(0, instrument.AcousticBass())
    low_note = note.Note(36)
    low_note.volume.velocity = 70
    bass.insert(0.0, low_note)
    next_low_note = note.Note(38)
    next_low_note.volume.velocity = 72
    bass.insert(1.0, next_low_note)

    score.insert(0, melody)
    score.insert(0, bass)
    _write_stream(filepath, score)


def _create_chord_midi(filepath):
    """Create a MIDI fixture with simultaneous notes."""
    score = stream.Stream()
    score.append(tempo.MetronomeMark(number=120))
    part = stream.Part()

    first = chord.Chord([60, 64, 67])
    first.volume.velocity = 80
    part.insert(0.0, first)

    second_low = note.Note(55)
    second_low.volume.velocity = 100
    part.insert(1.0, second_low)
    second_high = note.Note(72)
    second_high.volume.velocity = 60
    part.insert(1.0, second_high)

    score.insert(0, part)
    _write_stream(filepath, score)


def test_load_midi_returns_melody_curve(simple_midi_file):
    curve = load_midi(simple_midi_file)

    assert curve is not None
    assert curve.name == "simple_scale"
    assert curve.filepath == simple_midi_file
    assert len(curve.raw_notes) == 8
    assert curve.raw_notes[0].pitch == 60
    assert curve.raw_notes[0].timestamp == pytest.approx(0.0)
    assert curve.raw_notes[0].velocity == 80


def test_load_midi_converts_offsets_to_seconds(simple_midi_file):
    curve = load_midi(simple_midi_file)

    assert curve is not None
    assert curve.raw_notes[1].timestamp == pytest.approx(0.5)


def test_load_midi_sets_label_none(simple_midi_file):
    curve = load_midi(simple_midi_file)

    assert curve is not None
    assert curve.label is None


def test_load_midi_files_batch(simple_midi_file, sibling_midi_file):
    curves = load_midi_files([simple_midi_file, sibling_midi_file])

    assert len(curves) == 2
    assert curves[0].name == "simple_scale"
    assert curves[1].name == "shifted"


def test_load_midi_rejects_non_midi(tmp_path):
    bad_file = tmp_path / "not_midi.txt"
    bad_file.write_text("hello", encoding="utf-8")

    curve = load_midi(str(bad_file))

    assert curve is None


def test_load_midi_returns_none_for_missing_file(tmp_path):
    missing_file = tmp_path / "missing.mid"

    curve = load_midi(str(missing_file))

    assert curve is None


def test_inspect_midi_tracks_returns_track_summaries(tmp_path):
    filepath = tmp_path / "multi.mid"
    _create_multi_track_midi(str(filepath))

    tracks = inspect_midi_tracks(str(filepath))

    assert len(tracks) == 2
    assert [track.note_count for track in tracks] == [2, 2]
    assert tracks[0].pitch_min is not None
    assert tracks[0].pitch_max is not None


def test_load_midi_selects_track_index(tmp_path):
    filepath = tmp_path / "multi.mid"
    _create_multi_track_midi(str(filepath))

    melody_curve = load_midi(str(filepath), track_index=0)
    bass_curve = load_midi(str(filepath), track_index=1)

    assert melody_curve is not None
    assert bass_curve is not None
    assert {note.pitch for note in melody_curve.raw_notes} == {72, 74}
    assert {note.pitch for note in bass_curve.raw_notes} == {36, 38}


def test_load_midi_highest_mode_collapses_simultaneous_notes(tmp_path):
    filepath = tmp_path / "chord.mid"
    _create_chord_midi(str(filepath))

    all_curve = load_midi(str(filepath), extraction_mode="all")
    highest_curve = load_midi(str(filepath), extraction_mode="highest")

    assert all_curve is not None
    assert highest_curve is not None
    assert len(all_curve.raw_notes) == 5
    assert [note.pitch for note in highest_curve.raw_notes] == [67, 72]


def test_strongest_mode_prefers_loudest_simultaneous_note():
    notes = [
        Note(0.0, 60, 80),
        Note(0.0, 67, 70),
        Note(1.0, 55, 100),
        Note(1.0, 72, 60),
    ]

    collapsed = _collapse_simultaneous_notes(notes, mode="strongest")

    assert [note.pitch for note in collapsed] == [60, 55]


def test_load_midi_rejects_invalid_track_and_mode(simple_midi_file):
    assert load_midi(simple_midi_file, track_index=99) is None
    with pytest.raises(ValueError):
        load_midi(simple_midi_file, extraction_mode="invalid")
