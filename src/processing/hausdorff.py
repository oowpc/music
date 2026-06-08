import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    """Return a float point array with shape (N, D), rejecting empty sets."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D point array")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    return array


def hausdorff_standard(a_points: np.ndarray, b_points: np.ndarray) -> float:
    """Return standard Hausdorff distance between two point arrays.

    Both inputs must have shape (N, D) and (M, D). The return value is the
    symmetric Hausdorff distance as a float.
    """
    a_points = _validate_points(a_points, "a_points")
    b_points = _validate_points(b_points, "b_points")
    return float(
        max(
            directed_hausdorff(a_points, b_points)[0],
            directed_hausdorff(b_points, a_points)[0],
        )
    )


def hausdorff_modified(a_points: np.ndarray, b_points: np.ndarray) -> float:
    """Return modified Hausdorff distance using mean nearest-neighbor distance.

    Both inputs must have shape (N, D) and (M, D). The return value is the
    symmetric modified Hausdorff distance as a float.
    """
    a_points = _validate_points(a_points, "a_points")
    b_points = _validate_points(b_points, "b_points")

    b_tree = KDTree(b_points)
    distances_ab, _ = b_tree.query(a_points)

    a_tree = KDTree(a_points)
    distances_ba, _ = a_tree.query(b_points)

    return float(max(np.mean(distances_ab), np.mean(distances_ba)))


def frechet_discrete(a_points: np.ndarray, b_points: np.ndarray) -> float:
    """Return discrete Frechet distance between two ordered point arrays.

    Both inputs must have shape (N, D) and (M, D). Unlike Hausdorff distance,
    this metric respects point order along the two curves.
    """
    a_points = _validate_points(a_points, "a_points")
    b_points = _validate_points(b_points, "b_points")

    rows, cols = len(a_points), len(b_points)
    costs = np.empty((rows, cols), dtype=np.float64)

    for row in range(rows):
        for col in range(cols):
            distance = float(np.linalg.norm(a_points[row] - b_points[col]))
            if row == 0 and col == 0:
                costs[row, col] = distance
            elif row == 0:
                costs[row, col] = max(costs[row, col - 1], distance)
            elif col == 0:
                costs[row, col] = max(costs[row - 1, col], distance)
            else:
                costs[row, col] = max(
                    min(
                        costs[row - 1, col],
                        costs[row - 1, col - 1],
                        costs[row, col - 1],
                    ),
                    distance,
                )

    return float(costs[-1, -1])


def dtw_distance(a_points: np.ndarray, b_points: np.ndarray) -> float:
    """Return normalized Dynamic Time Warping distance between ordered curves.

    The accumulated path cost is divided by the warping path length, which keeps
    distances comparable across melodies with different point counts.
    """
    a_points = _validate_points(a_points, "a_points")
    b_points = _validate_points(b_points, "b_points")

    rows, cols = len(a_points), len(b_points)
    costs = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    lengths = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    costs[0, 0] = 0.0

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            step_distance = float(np.linalg.norm(a_points[row - 1] - b_points[col - 1]))
            candidates = [
                (costs[row - 1, col], lengths[row - 1, col]),
                (costs[row, col - 1], lengths[row, col - 1]),
                (costs[row - 1, col - 1], lengths[row - 1, col - 1]),
            ]
            previous_cost, previous_length = min(candidates, key=lambda item: (item[0], item[1]))
            costs[row, col] = previous_cost + step_distance
            lengths[row, col] = previous_length + 1

    path_length = int(lengths[rows, cols])
    return float(costs[rows, cols] / path_length) if path_length else 0.0
