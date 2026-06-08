import numpy as np

from src.analysis.method_comparison import compare_genre_methods
from src.models.melody_curve import MelodyCurve


def _curve(name: str, label: str, points) -> MelodyCurve:
    """Create a labeled normalized curve for method-comparison tests."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        label=label,
        points=np.array(points, dtype=float),
    )


def test_compare_genre_methods_returns_metrics_for_selected_methods():
    curves = [
        _curve("pop-a", "pop", [(0.0, 0.1, 0.7), (0.5, 0.2, 0.7), (1.0, 0.3, 0.7)]),
        _curve("pop-b", "pop", [(0.0, 0.1, 0.7), (0.5, 0.22, 0.7), (1.0, 0.31, 0.7)]),
        _curve("rock-a", "rock", [(0.0, 0.8, 0.7), (0.5, 0.7, 0.7), (1.0, 0.6, 0.7)]),
        _curve("rock-b", "rock", [(0.0, 0.81, 0.7), (0.5, 0.72, 0.7), (1.0, 0.62, 0.7)]),
    ]

    results = compare_genre_methods(curves, methods=["standard", "modified", "frechet", "dtw"], k=1)

    assert [result.method for result in results] == ["standard", "modified", "frechet", "dtw"]
    assert all(result.sample_count == 4 for result in results)
    assert all(result.genre_count == 2 for result in results)
    assert all(result.separation_ratio > 1.0 for result in results)
    assert all(result.knn_accuracy == 1.0 for result in results)


def test_compare_genre_methods_ignores_unlabeled_curves():
    curves = [
        _curve("a", "pop", [(0.0, 0.1, 0.7), (1.0, 0.2, 0.7)]),
        _curve("b", None, [(0.0, 0.9, 0.7), (1.0, 0.8, 0.7)]),
    ]

    assert compare_genre_methods(curves, methods=["dtw"], k=1) == []
