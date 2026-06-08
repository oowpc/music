import numpy as np
import pytest

from src.analysis.distance_matrix import build_matrix, extend_matrix
from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.processing.normalization import normalize_minmax


def make_curve(name, notes_data):
    """Create a test curve from tuples of (time, pitch, velocity)."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[Note(timestamp, pitch, velocity) for timestamp, pitch, velocity in notes_data],
    )


def test_build_matrix_square_symmetric():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    c2 = make_curve("b", [(0.0, 62, 70), (1.0, 67, 90)])
    c3 = make_curve("c", [(0.0, 72, 110), (0.5, 75, 120)])
    curves = [c1, c2, c3]
    normalize_minmax(curves)

    matrix = build_matrix(curves, method="standard")

    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == pytest.approx(0.0)
    assert matrix[1, 1] == pytest.approx(0.0)
    assert matrix[2, 2] == pytest.approx(0.0)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)


def test_build_matrix_modified_method():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    c2 = make_curve("b", [(0.0, 62, 70), (1.0, 67, 90)])
    curves = [c1, c2]
    normalize_minmax(curves)

    matrix = build_matrix(curves, method="modified")

    assert matrix.shape == (2, 2)
    assert matrix[0, 1] > 0.0


def test_build_matrix_frechet_method():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 90)])
    c2 = make_curve("b", [(0.0, 67, 90), (1.0, 64, 100), (2.0, 60, 80)])
    curves = [c1, c2]
    normalize_minmax(curves)

    matrix = build_matrix(curves, method="frechet")

    assert matrix.shape == (2, 2)
    assert matrix[0, 1] > 0.0
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)


def test_build_matrix_single_curve():
    curve = make_curve("solo", [(0.0, 60, 80)])
    normalize_minmax([curve])

    matrix = build_matrix([curve], method="standard")

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 0.0


def test_build_matrix_empty_list():
    assert build_matrix([], method="standard").shape == (0, 0)


def test_build_matrix_rejects_unknown_method():
    with pytest.raises(ValueError):
        build_matrix([], method="unknown")


def test_extend_matrix_preserves_existing_block():
    c1 = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    c2 = make_curve("b", [(0.0, 62, 70), (1.0, 67, 90)])
    c3 = make_curve("c", [(0.0, 72, 110), (0.5, 75, 120)])
    curves = [c1, c2]
    normalize_minmax(curves)
    original = build_matrix(curves, method="standard")

    curves.append(c3)
    normalize_minmax(curves)
    extended = extend_matrix(original, curves, previous_count=2, method="standard")
    rebuilt = build_matrix(curves, method="standard")

    np.testing.assert_allclose(extended[:2, :2], original)
    np.testing.assert_allclose(extended, rebuilt)


def test_extend_matrix_validates_shape():
    curve = make_curve("a", [(0.0, 60, 80)])
    normalize_minmax([curve])

    with pytest.raises(ValueError):
        extend_matrix(np.zeros((2, 2)), [curve], previous_count=1)
