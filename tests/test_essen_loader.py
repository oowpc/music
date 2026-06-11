"""Tests for src.io.essen_loader."""

import os
import tempfile

from src.io.essen_loader import (
    _write_midi,
    download_essen,
    get_essen_summary,
    load_essen_collection,
)
from src.models.melody_curve import MelodyCurve


def test_load_essen_collection_empty_dir(tmp_path):
    """Return an empty list when no MIDI files exist."""
    curves = load_essen_collection(str(tmp_path))
    assert curves == []


def test_load_essen_from_small_midi_set(tmp_path):
    """Load curves from a directory tree, verifying labels match folder names."""
    # Create two region folders, each with a MIDI file
    deutsch_dir = os.path.join(tmp_path, "deutsch")
    chinese_dir = os.path.join(tmp_path, "chinese")
    os.makedirs(deutsch_dir)
    os.makedirs(chinese_dir)

    _write_midi(os.path.join(deutsch_dir, "song1.mid"), [60, 62, 64, 65, 67],
                bpm=120.0)
    _write_midi(os.path.join(deutsch_dir, "song2.mid"), [62, 64, 65, 67, 69],
                bpm=120.0)
    _write_midi(os.path.join(chinese_dir, "song3.mid"), [60, 62, 64, 67, 69],
                bpm=120.0)

    curves = load_essen_collection(str(tmp_path))

    assert len(curves) == 3
    labels = {curve.label for curve in curves}
    assert labels == {"deutsch", "chinese"}

    # All curves should have a color assigned
    for curve in curves:
        assert curve.color.startswith("#")
        assert len(curve.color) == 7

    # Curves with the same label share the same color
    deutsch_curves = [c for c in curves if c.label == "deutsch"]
    assert len(deutsch_curves) == 2
    assert deutsch_curves[0].color == deutsch_curves[1].color
    # Different labels get different colors
    chinese_curve = next(c for c in curves if c.label == "chinese")
    assert chinese_curve.color != deutsch_curves[0].color


def test_load_essen_collection_max_files(tmp_path):
    """Respect the max_files limit."""
    region_dir = os.path.join(tmp_path, "deutsch")
    os.makedirs(region_dir)
    _write_midi(os.path.join(region_dir, "song1.mid"), [60, 62, 64], bpm=120.0)
    _write_midi(os.path.join(region_dir, "song2.mid"), [62, 64, 65], bpm=120.0)
    _write_midi(os.path.join(region_dir, "song3.mid"), [64, 67, 69], bpm=120.0)

    curves = load_essen_collection(str(tmp_path), max_files=2)

    assert len(curves) == 2


def test_get_essen_summary(tmp_path):
    """Verify per-label counts."""
    region_a_dir = os.path.join(tmp_path, "region_a")
    region_b_dir = os.path.join(tmp_path, "region_b")
    os.makedirs(region_a_dir)
    os.makedirs(region_b_dir)

    _write_midi(os.path.join(region_a_dir, "a1.mid"), [60, 62, 64], bpm=120.0)
    _write_midi(os.path.join(region_a_dir, "a2.mid"), [62, 64, 65], bpm=120.0)
    _write_midi(os.path.join(region_b_dir, "b1.mid"), [64, 67, 69], bpm=120.0)

    curves = load_essen_collection(str(tmp_path))
    summary = get_essen_summary(curves)

    assert summary == {"region_a": 2, "region_b": 1}


def test_download_essen_creates_synthetic_on_fresh_dir(tmp_path):
    """Download (or generate) Essen data into a clean directory."""
    target = os.path.join(tmp_path, "essen_test")
    result = download_essen(target)

    assert result == target
    assert os.path.isdir(target)

    curves = load_essen_collection(target)
    assert len(curves) >= 1
    # At least one label exists
    assert get_essen_summary(curves)


def test_download_essen_skips_when_midi_exists(tmp_path):
    """Do not re-download when MIDI files are already present."""
    target = os.path.join(tmp_path, "essen_test")
    os.makedirs(target)

    # Place a dummy MIDI file to simulate existing data
    _write_midi(os.path.join(target, "existing.mid"), [60, 62, 64], bpm=120.0)

    result = download_essen(target)
    assert result == target

    # Only the original file should exist (no synthetic data generated)
    curves = load_essen_collection(target)
    assert len(curves) == 1
    assert curves[0].name == "existing"
