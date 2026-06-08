import numpy as np

from src.models.melody_curve import MelodyCurve
from src.processing.hausdorff import (
    dtw_distance,
    frechet_discrete,
    hausdorff_modified,
    hausdorff_standard,
)


DISTANCE_FUNCTIONS = {
    "standard": hausdorff_standard,
    "modified": hausdorff_modified,
    "frechet": frechet_discrete,
    "dtw": dtw_distance,
}


def build_matrix(curves: list[MelodyCurve], method: str = "standard") -> np.ndarray:
    """Compute an NxN symmetric distance matrix for melody curves.

    Curves are expected to have normalized ``points`` arrays with shape (N, 3).
    Missing or empty point arrays are skipped and leave zero distances.
    """
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance method: {method}")

    curve_count = len(curves)
    matrix = np.zeros((curve_count, curve_count), dtype=np.float64)
    distance_function = DISTANCE_FUNCTIONS[method]

    for row in range(curve_count):
        for col in range(row + 1, curve_count):
            left = curves[row].points
            right = curves[col].points
            if left is None or right is None or len(left) == 0 or len(right) == 0:
                continue
            distance = distance_function(left, right)
            matrix[row, col] = distance
            matrix[col, row] = distance

    return matrix


def extend_matrix(
    existing_matrix: np.ndarray | None,
    curves: list[MelodyCurve],
    previous_count: int,
    method: str = "standard",
) -> np.ndarray:
    """Extend a distance matrix after appending curves.

    ``existing_matrix`` must contain distances for the first
    ``previous_count`` curves. The function preserves that old block and only
    computes distances for pairs where at least one index belongs to a newly
    appended curve.
    """
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance method: {method}")

    curve_count = len(curves)
    if previous_count < 0 or previous_count > curve_count:
        raise ValueError("previous_count must be between 0 and len(curves)")

    if existing_matrix is None or previous_count == 0:
        return build_matrix(curves, method=method)

    existing_matrix = np.asarray(existing_matrix, dtype=np.float64)
    if existing_matrix.shape != (previous_count, previous_count):
        raise ValueError("existing_matrix shape must match previous_count")

    matrix = np.zeros((curve_count, curve_count), dtype=np.float64)
    matrix[:previous_count, :previous_count] = existing_matrix
    distance_function = DISTANCE_FUNCTIONS[method]

    for row in range(previous_count, curve_count):
        for col in range(row):
            left = curves[row].points
            right = curves[col].points
            if left is None or right is None or len(left) == 0 or len(right) == 0:
                continue
            distance = distance_function(left, right)
            matrix[row, col] = distance
            matrix[col, row] = distance

    return matrix
