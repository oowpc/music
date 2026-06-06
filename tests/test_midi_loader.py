import pytest

from src.io.midi_loader import load_midi, load_midi_files


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
