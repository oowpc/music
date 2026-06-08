import numpy as np

from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.processing.normalization import normalize_minmax, normalize_zscore


def make_curve(name, notes_data):
    """Create a test curve from tuples of (time, pitch, velocity)."""
    raw_notes = [Note(timestamp, pitch, velocity) for timestamp, pitch, velocity in notes_data]
    return MelodyCurve(name=name, filepath=f"/fake/{name}.mid", raw_notes=raw_notes)


def test_normalize_minmax_maps_to_unit_cube():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 60)])
    c2 = make_curve("b", [(0.0, 50, 50), (3.0, 72, 127)])

    normalize_minmax([c1, c2])

    for curve in [c1, c2]:
        assert curve.points is not None
        assert np.all(curve.points >= 0.0)
        assert np.all(curve.points <= 1.0)


def test_normalize_minmax_same_curve_zero_variance():
    curve = make_curve("single", [(1.0, 60, 80)])

    normalize_minmax([curve])

    assert curve.points is not None
    assert curve.points.shape == (1, 3)
    np.testing.assert_allclose(curve.points, [[0.0, 60 / 127.0, 80 / 127.0]])


def test_normalize_zscore_maps_to_zero_mean():
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 60)])

    normalize_zscore([curve])

    assert curve.points is not None
    np.testing.assert_allclose(np.mean(curve.points, axis=0), 0.0, atol=1e-10)


def test_normalize_minmax_uses_per_curve_time_scale():
    c1 = make_curve("a", [(0.0, 60, 80), (2.0, 72, 100)])
    c2 = make_curve("b", [(0.0, 64, 90), (1.0, 67, 80)])

    normalize_minmax([c1, c2])

    span1 = np.max(c1.points[:, 0]) - np.min(c1.points[:, 0])
    span2 = np.max(c2.points[:, 0]) - np.min(c2.points[:, 0])
    assert span1 == 1.0
    assert span2 == 1.0


def test_normalize_minmax_existing_curve_stable_when_new_curve_added():
    c1 = make_curve("a", [(0.0, 60, 80), (2.0, 72, 100)])
    c2 = make_curve("b", [(0.0, 36, 20), (10.0, 84, 127)])

    normalize_minmax([c1])
    original = c1.points.copy()
    normalize_minmax([c1, c2])

    np.testing.assert_allclose(c1.points, original)


def test_normalize_empty_curves_list():
    normalize_minmax([])
    normalize_zscore([])


def test_normalize_empty_curve_gets_empty_points():
    curve = make_curve("empty", [])

    normalize_minmax([curve])

    assert curve.points is not None
    assert curve.points.shape == (0, 3)
