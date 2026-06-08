import math

import numpy as np

from src.analysis.genre_analysis import genre_distance_matrix, knn_genre_classification, labeled_distance_data
from src.models.melody_curve import MelodyCurve


def _curve(name: str, label: str | None) -> MelodyCurve:
    """Create a labeled curve for genre-analysis tests."""
    return MelodyCurve(name=name, filepath=f"/fake/{name}.mid", label=label)


def test_labeled_distance_data_filters_unlabeled_curves():
    matrix = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.8],
            [0.9, 0.8, 0.0],
        ]
    )
    curves = [_curve("a", "pop"), _curve("b", None), _curve("c", "rock")]

    result = labeled_distance_data(matrix, curves)

    assert result.labels == ["pop", "rock"]
    assert result.names == ["a", "c"]
    assert result.original_indices == [0, 2]
    assert result.matrix.tolist() == [[0.0, 0.9], [0.9, 0.0]]


def test_genre_distance_matrix_averages_within_and_between_labels():
    matrix = np.array(
        [
            [0.0, 0.2, 1.0, 1.2, 2.0],
            [0.2, 0.0, 0.8, 1.0, 2.2],
            [1.0, 0.8, 0.0, 0.4, 1.6],
            [1.2, 1.0, 0.4, 0.0, 1.4],
            [2.0, 2.2, 1.6, 1.4, 0.0],
        ]
    )
    curves = [
        _curve("a1", "pop"),
        _curve("a2", "pop"),
        _curve("b1", "rock"),
        _curve("b2", "rock"),
        _curve("c1", "jazz"),
    ]

    result = genre_distance_matrix(matrix, curves)

    assert result.labels == ["jazz", "pop", "rock"]
    assert result.counts == {"jazz": 1, "pop": 2, "rock": 2}
    assert math.isnan(result.matrix[0, 0])
    assert result.matrix[1, 1] == 0.2
    assert result.matrix[2, 2] == 0.4
    assert result.matrix[1, 2] == 1.0
    assert result.matrix[2, 1] == 1.0


def test_knn_genre_classification_runs_leave_one_out_for_arbitrary_labels():
    matrix = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 0.9, 1.0],
            [1.0, 0.9, 0.0, 0.2],
            [1.1, 1.0, 0.2, 0.0],
        ]
    )
    curves = [
        _curve("a1", "ambient"),
        _curve("a2", "ambient"),
        _curve("b1", "metal"),
        _curve("b2", "metal"),
    ]

    result = knn_genre_classification(matrix, curves, k=1)

    assert result.k == 1
    assert result.labels == ["ambient", "metal"]
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]
