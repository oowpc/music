from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS, build_matrix
from src.analysis.genre_analysis import genre_distance_matrix, knn_genre_classification
from src.models.melody_curve import MelodyCurve


DEFAULT_METHODS = ["standard", "modified", "frechet", "dtw"]

METHOD_DISPLAY_NAMES = {
    "standard": "标准 Hausdorff",
    "modified": "Modified Hausdorff",
    "frechet": "离散 Fréchet",
    "dtw": "DTW",
}


@dataclass(frozen=True)
class MethodComparisonResult:
    """Genre-separation summary for one distance method."""

    method: str
    display_name: str
    within_avg: float | None
    between_avg: float | None
    separation_ratio: float | None
    knn_accuracy: float | None
    macro_f1: float | None
    sample_count: int
    genre_count: int


def compare_genre_methods(
    curves: list[MelodyCurve],
    methods: list[str] | None = None,
    k: int = 5,
) -> list[MethodComparisonResult]:
    """Compare distance methods by genre separation and leave-one-out KNN."""
    selected_methods = methods or DEFAULT_METHODS
    for method in selected_methods:
        if method not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Unsupported distance method: {method}")

    labeled_curves = [
        curve
        for curve in curves
        if curve.label and curve.points is not None and len(curve.points) > 0
    ]
    if len(labeled_curves) < 2:
        return []

    results = []
    for method in selected_methods:
        matrix = build_matrix(labeled_curves, method=method)
        genre_result = genre_distance_matrix(matrix, labeled_curves)
        within_avg, between_avg, separation_ratio = _genre_separation(genre_result.matrix)

        knn_result = knn_genre_classification(matrix, labeled_curves, k=k)
        if knn_result is None:
            knn_accuracy = None
            macro_f1 = None
        else:
            knn_accuracy = float(knn_result.metrics["accuracy"])
            macro_f1 = float(knn_result.metrics["macro_f1"])

        labels = {str(curve.label) for curve in labeled_curves}
        results.append(
            MethodComparisonResult(
                method=method,
                display_name=METHOD_DISPLAY_NAMES.get(method, method),
                within_avg=within_avg,
                between_avg=between_avg,
                separation_ratio=separation_ratio,
                knn_accuracy=knn_accuracy,
                macro_f1=macro_f1,
                sample_count=len(labeled_curves),
                genre_count=len(labels),
            )
        )
    return results


def _genre_separation(matrix: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Return within average, between average, and between/within ratio."""
    if matrix.size == 0:
        return None, None, None

    within_values = [float(matrix[index, index]) for index in range(matrix.shape[0]) if np.isfinite(matrix[index, index])]
    between_values = [
        float(matrix[row, col])
        for row in range(matrix.shape[0])
        for col in range(row + 1, matrix.shape[1])
        if np.isfinite(matrix[row, col])
    ]
    within_avg = float(np.mean(within_values)) if within_values else None
    between_avg = float(np.mean(between_values)) if between_values else None
    if within_avg is None or between_avg is None or within_avg <= 0:
        return within_avg, between_avg, None
    return within_avg, between_avg, between_avg / within_avg
