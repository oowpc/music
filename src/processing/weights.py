import numpy as np

from src.models.melody_curve import MelodyCurve


def apply_dimension_weights(
    curves: list[MelodyCurve],
    weights: tuple[float, float, float],
) -> None:
    """Multiply normalized curve points by time, pitch, and velocity weights."""
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (3,):
        raise ValueError("weights must contain exactly three values")
    if np.any(weight_array < 0):
        raise ValueError("weights must be non-negative")

    for curve in curves:
        if curve.points is None:
            continue
        if curve.points.ndim != 2 or curve.points.shape[1] != 3:
            raise ValueError("curve points must have shape (N, 3)")
        curve.points = curve.points * weight_array
