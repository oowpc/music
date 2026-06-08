from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analysis.knn_classifier import KNNPrediction, evaluate_predictions, leave_one_out_knn
from src.models.melody_curve import MelodyCurve


@dataclass(frozen=True)
class LabeledDistanceData:
    """Distance matrix subset that only contains curves with genre labels."""

    matrix: np.ndarray
    labels: list[str]
    names: list[str]
    original_indices: list[int]


@dataclass(frozen=True)
class GenreDistanceResult:
    """Average pairwise distances grouped by genre label."""

    labels: list[str]
    matrix: np.ndarray
    counts: dict[str, int]


@dataclass(frozen=True)
class KNNGenreResult:
    """Leave-one-out KNN classification result for labeled curves."""

    k: int
    labels: list[str]
    predictions: list[KNNPrediction]
    metrics: dict


def labeled_distance_data(matrix: np.ndarray, curves: list[MelodyCurve]) -> LabeledDistanceData:
    """Return the distance submatrix for curves that have a non-empty label."""
    full_matrix = np.asarray(matrix, dtype=np.float64)
    if full_matrix.shape != (len(curves), len(curves)):
        raise ValueError("matrix shape must match curve count")

    indices = [index for index, curve in enumerate(curves) if curve.label]
    labels = [str(curves[index].label) for index in indices]
    names = [curves[index].name for index in indices]
    subset = full_matrix[np.ix_(indices, indices)] if indices else np.zeros((0, 0))
    return LabeledDistanceData(subset, labels, names, indices)


def genre_distance_matrix(matrix: np.ndarray, curves: list[MelodyCurve]) -> GenreDistanceResult | None:
    """Compute the average distance matrix between arbitrary genre labels.

    Diagonal cells average all distinct pairs within one genre. If a genre only
    has one sample, the diagonal value is NaN because no within-genre pair
    exists.
    """
    labeled = labeled_distance_data(matrix, curves)
    if not labeled.labels:
        return None

    genre_labels = sorted(set(labeled.labels))
    counts = {label: labeled.labels.count(label) for label in genre_labels}
    result = np.full((len(genre_labels), len(genre_labels)), np.nan, dtype=np.float64)

    for row, left_label in enumerate(genre_labels):
        left_indices = [index for index, label in enumerate(labeled.labels) if label == left_label]
        for col, right_label in enumerate(genre_labels):
            right_indices = [index for index, label in enumerate(labeled.labels) if label == right_label]
            distances = []
            for left_index in left_indices:
                for right_index in right_indices:
                    if left_label == right_label and left_index >= right_index:
                        continue
                    distances.append(float(labeled.matrix[left_index, right_index]))
            if distances:
                result[row, col] = float(np.mean(distances))

    return GenreDistanceResult(genre_labels, result, counts)


def knn_genre_classification(
    matrix: np.ndarray,
    curves: list[MelodyCurve],
    k: int = 5,
) -> KNNGenreResult | None:
    """Run leave-one-out KNN on labeled curves and return metrics."""
    if k <= 0:
        raise ValueError("k must be positive")

    labeled = labeled_distance_data(matrix, curves)
    if len(labeled.labels) < 2 or len(set(labeled.labels)) < 2:
        return None

    effective_k = min(k, len(labeled.labels) - 1)
    predictions = leave_one_out_knn(labeled.matrix, labeled.labels, k=effective_k)
    metrics = evaluate_predictions(predictions, label_order=sorted(set(labeled.labels)))
    return KNNGenreResult(effective_k, sorted(set(labeled.labels)), predictions, metrics)
