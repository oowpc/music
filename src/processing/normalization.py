import numpy as np

from src.models.melody_curve import MelodyCurve


def _curve_points(curve: MelodyCurve) -> np.ndarray:
    """Return raw curve notes as an array with shape (N, 3)."""
    return np.array(
        [(note.timestamp, note.pitch, note.velocity) for note in curve.raw_notes],
        dtype=np.float64,
    )


def normalize_minmax(curves: list[MelodyCurve]) -> None:
    """Normalize each curve to [0, 1]^3 using stable musical ranges.

    Input curves contain raw notes with (seconds, MIDI pitch, MIDI velocity).
    The function mutates each curve's ``points`` field to an array with shape
    (N, 3). Time is normalized by each curve's own duration, while pitch and
    velocity use fixed MIDI ranges divided by 127. Empty curves receive an
    empty (0, 3) array.
    """
    if not curves:
        return

    for curve in curves:
        points = _curve_points(curve)
        if points.size == 0:
            curve.points = points.reshape(0, 3)
            continue

        normalized = np.zeros_like(points, dtype=np.float64)
        time_min = points[:, 0].min()
        time_range = points[:, 0].max() - time_min
        normalized[:, 0] = (points[:, 0] - time_min) / time_range if time_range > 0 else 0.0
        normalized[:, 1] = points[:, 1] / 127.0
        normalized[:, 2] = points[:, 2] / 127.0
        curve.points = normalized


def normalize_zscore(curves: list[MelodyCurve]) -> None:
    """Normalize each curve to zero mean and unit standard deviation.

    Input curves contain raw notes with (seconds, MIDI pitch, MIDI velocity).
    The function mutates each curve's ``points`` field to an array with shape
    (N, 3). Dimensions with zero variance are set to 0.
    """
    if not curves:
        return

    for curve in curves:
        points = _curve_points(curve)
        if points.size == 0:
            curve.points = points.reshape(0, 3)
            continue

        means = points.mean(axis=0)
        stds = points.std(axis=0)
        normalized = np.zeros_like(points, dtype=np.float64)
        for axis in range(3):
            if stds[axis] > 0:
                normalized[:, axis] = (points[:, axis] - means[axis]) / stds[axis]
            else:
                normalized[:, axis] = 0.0
        curve.points = normalized
