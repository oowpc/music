from dataclasses import dataclass

import numpy as np

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS
from src.models.melody_curve import MelodyCurve


@dataclass(frozen=True)
class SegmentMatch:
    """Best matching segment pair between two melody curves."""

    left_start: float
    left_end: float
    right_start: float
    right_end: float
    distance: float
    left_points: int
    right_points: int


def slice_points_by_time(
    points: np.ndarray,
    start: float,
    end: float,
    time_axis: int = 0,
) -> np.ndarray:
    """Return points with normalized time in [start, end]."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if points.shape[0] == 0:
        return points.reshape(0, points.shape[1])

    mask = (points[:, time_axis] >= start) & (points[:, time_axis] <= end)
    return points[mask]


def sliding_windows(
    points: np.ndarray,
    window_size: float,
    step_size: float,
    min_points: int = 2,
) -> list[tuple[float, float, np.ndarray]]:
    """Generate normalized-time sliding windows over a point array.

    Returns a list of ``(start, end, window_points)`` tuples. Windows with fewer
    than ``min_points`` are skipped.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    if points.shape[0] == 0:
        return []

    max_time = float(np.max(points[:, 0]))
    windows = []
    start = float(np.min(points[:, 0]))
    stop = max(max_time - window_size, start)

    while start <= stop + 1e-12:
        end = min(start + window_size, max_time)
        segment = _localize_segment_time(slice_points_by_time(points, start, end), start, end)
        if len(segment) >= min_points:
            windows.append((start, end, segment))
        start += step_size

    if not windows and len(points) >= min_points:
        windows.append((float(np.min(points[:, 0])), max_time, points))
    return windows


def _localize_segment_time(points: np.ndarray, start: float, end: float) -> np.ndarray:
    """Return a copy whose time axis is normalized within the segment window."""
    localized = np.array(points, dtype=np.float64, copy=True)
    if localized.shape[0] == 0:
        return localized
    duration = end - start
    localized[:, 0] = (localized[:, 0] - start) / duration if duration > 0 else 0.0
    return localized


def find_best_segment_match(
    left: MelodyCurve,
    right: MelodyCurve,
    method: str = "modified",
    window_size: float = 0.25,
    step_size: float = 0.05,
    min_points: int = 2,
) -> SegmentMatch | None:
    """Find the lowest-distance sliding-window segment pair for two curves."""
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance method: {method}")
    if left.points is None or right.points is None:
        return None

    left_windows = sliding_windows(left.points, window_size, step_size, min_points=min_points)
    right_windows = sliding_windows(right.points, window_size, step_size, min_points=min_points)
    if not left_windows or not right_windows:
        return None

    distance_function = DISTANCE_FUNCTIONS[method]
    best: SegmentMatch | None = None
    for left_start, left_end, left_points in left_windows:
        for right_start, right_end, right_points in right_windows:
            distance = distance_function(left_points, right_points)
            if best is None or distance < best.distance:
                best = SegmentMatch(
                    left_start=left_start,
                    left_end=left_end,
                    right_start=right_start,
                    right_end=right_end,
                    distance=float(distance),
                    left_points=len(left_points),
                    right_points=len(right_points),
                )
    return best
