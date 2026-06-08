import numpy as np
import pytest

from src.models.melody_curve import MelodyCurve
from src.processing.weights import apply_dimension_weights


def test_apply_dimension_weights_multiplies_points():
    curve = MelodyCurve(name="a", filepath="/fake/a.mid")
    curve.points = np.array([[0.5, 0.25, 0.75]])

    apply_dimension_weights([curve], (2.0, 3.0, 0.5))

    np.testing.assert_allclose(curve.points, [[1.0, 0.75, 0.375]])


def test_apply_dimension_weights_rejects_bad_weights():
    curve = MelodyCurve(name="a", filepath="/fake/a.mid")
    curve.points = np.array([[0.5, 0.25, 0.75]])

    with pytest.raises(ValueError):
        apply_dimension_weights([curve], (1.0, 2.0))

    with pytest.raises(ValueError):
        apply_dimension_weights([curve], (1.0, -1.0, 1.0))
