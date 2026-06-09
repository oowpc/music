import pytest
import numpy as np

from src.analysis.evaluation import evaluate, evaluate_distance_matrix, silhouette_score_precomputed
from src.models.melody_curve import MelodyCurve


def make_curve(name, label):
    """Create a test curve with an optional label."""
    return MelodyCurve(name=name, filepath=f"/fake/{name}.mid", label=label)


def test_evaluate_returns_metrics_with_labels():
    curves = [
        make_curve("a", "classical"),
        make_curve("b", "classical"),
        make_curve("c", "pop"),
        make_curve("d", "pop"),
    ]

    result = evaluate(curves, [1, 1, 2, 2])

    assert result["ari"] == pytest.approx(1.0)
    assert result["purity"] == pytest.approx(1.0)


def test_evaluate_returns_empty_without_labels():
    curves = [
        make_curve("a", None),
        make_curve("b", None),
    ]

    assert evaluate(curves, [1, 1]) == {}


def test_evaluate_mixed_labels():
    curves = [
        make_curve("a", "classical"),
        make_curve("b", None),
        make_curve("c", "pop"),
    ]

    assert evaluate(curves, [1, 1, 2]) == {}


def test_evaluate_returns_empty_for_single_true_label():
    curves = [
        make_curve("a", "classical"),
        make_curve("b", "classical"),
    ]

    assert evaluate(curves, [1, 1]) == {}


def test_silhouette_score_precomputed_returns_positive_for_separated_labels():
    matrix = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 0.9, 1.0],
            [1.0, 0.9, 0.0, 0.2],
            [1.1, 1.0, 0.2, 0.0],
        ]
    )

    score = silhouette_score_precomputed(matrix, ["pop", "pop", "rock", "rock"])

    assert score == pytest.approx(0.849624, abs=1e-6)


def test_evaluate_distance_matrix_returns_silhouette():
    curves = [
        make_curve("a", "pop"),
        make_curve("b", "pop"),
        make_curve("c", "rock"),
        make_curve("d", "rock"),
    ]
    matrix = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 0.9, 1.0],
            [1.0, 0.9, 0.0, 0.2],
            [1.1, 1.0, 0.2, 0.0],
        ]
    )

    result = evaluate_distance_matrix(curves, matrix)

    assert result["silhouette"] == pytest.approx(0.8496, abs=1e-4)
