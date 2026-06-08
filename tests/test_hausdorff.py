import numpy as np
import pytest

from src.processing.hausdorff import frechet_discrete, hausdorff_modified, hausdorff_standard


def test_hausdorff_identical_sets():
    points = np.array([[0.0, 0.5, 0.3], [1.0, 0.8, 0.2], [0.5, 0.3, 0.9]])

    assert hausdorff_standard(points, points.copy()) == pytest.approx(0.0, abs=1e-10)
    assert hausdorff_modified(points, points.copy()) == pytest.approx(0.0, abs=1e-10)
    assert frechet_discrete(points, points.copy()) == pytest.approx(0.0, abs=1e-10)


def test_hausdorff_different_sets_positive():
    left = np.array([[0.0, 0.0, 0.0]])
    right = np.array([[1.0, 1.0, 1.0]])

    assert hausdorff_standard(left, right) > 0.0


def test_hausdorff_modified_less_than_standard_with_outlier():
    rng = np.random.default_rng(42)
    left = rng.random((20, 3))
    right = rng.random((20, 3))
    right[-1] = np.array([5.0, 5.0, 5.0])

    assert hausdorff_modified(left, right) < hausdorff_standard(left, right)


def test_hausdorff_symmetry():
    left = np.array([[0.0, 0.5, 0.3], [1.0, 0.8, 0.2]])
    right = np.array([[0.2, 0.3, 0.7], [0.9, 0.6, 0.1]])

    assert hausdorff_standard(left, right) == pytest.approx(hausdorff_standard(right, left))
    assert hausdorff_modified(left, right) == pytest.approx(hausdorff_modified(right, left))


def test_hausdorff_single_point_each():
    left = np.array([[0.0, 0.0, 0.0]])
    right = np.array([[0.3, 0.4, 0.0]])

    assert hausdorff_standard(left, right) == pytest.approx(0.5)


def test_hausdorff_rejects_empty_sets():
    with pytest.raises(ValueError):
        hausdorff_standard(np.empty((0, 3)), np.array([[0.0, 0.0, 0.0]]))


def test_frechet_single_point_each():
    left = np.array([[0.0, 0.0, 0.0]])
    right = np.array([[0.3, 0.4, 0.0]])

    assert frechet_discrete(left, right) == pytest.approx(0.5)


def test_frechet_respects_curve_order():
    forward = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    reversed_curve = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    assert hausdorff_standard(forward, reversed_curve) == pytest.approx(0.0)
    assert frechet_discrete(forward, reversed_curve) == pytest.approx(2.0)


def test_frechet_rejects_empty_sets():
    with pytest.raises(ValueError):
        frechet_discrete(np.empty((0, 3)), np.array([[0.0, 0.0, 0.0]]))
