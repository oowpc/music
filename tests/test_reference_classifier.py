import numpy as np
import pytest

from src.analysis.reference_classifier import (
    ReferenceItem,
    classify_query_curve,
    resample_points,
)
from src.models.melody_curve import MelodyCurve


def test_resample_points_returns_fixed_count():
    points = np.array([[0.0, 0.1, 0.7], [0.5, 0.2, 0.7], [1.0, 0.3, 0.7]])

    sampled = resample_points(points, target_points=5)

    assert sampled.shape == (5, 3)
    assert sampled[0, 0] == pytest.approx(0.0)
    assert sampled[-1, 0] == pytest.approx(1.0)


def test_classify_query_curve_predicts_majority_label():
    references = [
        ReferenceItem("pop-a", "pop", "/fake/pop-a.mid", np.array([[0.0, 0.1, 0.7], [1.0, 0.2, 0.7]])),
        ReferenceItem("pop-b", "pop", "/fake/pop-b.mid", np.array([[0.0, 0.11, 0.7], [1.0, 0.21, 0.7]])),
        ReferenceItem("rock-a", "rock", "/fake/rock-a.mid", np.array([[0.0, 0.8, 0.7], [1.0, 0.7, 0.7]])),
    ]
    query = MelodyCurve(
        name="query",
        filepath="/fake/query.mid",
        points=np.array([[0.0, 0.12, 0.7], [1.0, 0.22, 0.7]]),
    )

    result = classify_query_curve(query, references, method="modified", k=3)

    assert result.predicted_label == "pop"
    assert [neighbor.label for neighbor in result.neighbors[:2]] == ["pop", "pop"]


def test_classify_query_curve_validates_inputs():
    query = MelodyCurve(name="query", filepath="/fake/query.mid", points=np.array([[0.0, 0.1, 0.7]]))

    with pytest.raises(ValueError):
        classify_query_curve(query, [], k=1)
    with pytest.raises(ValueError):
        classify_query_curve(query, [ReferenceItem("a", "pop", "/fake/a.mid", query.points)], k=0)
