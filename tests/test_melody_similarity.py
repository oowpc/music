import numpy as np
import pytest

from src.analysis.melody_similarity import (
    classify_similarity_level,
    compare_melodies,
    similarity_score,
)
from src.models.melody_curve import MelodyCurve


def _curve(name: str, points) -> MelodyCurve:
    """Create a prepared curve for melody similarity tests."""
    return MelodyCurve(name=name, filepath=f"/fake/{name}.mid", points=np.array(points, dtype=float))


def test_compare_melodies_detects_high_similarity():
    left = _curve("left", [(0.0, 0.1, 0.7), (0.5, 0.2, 0.7), (1.0, 0.3, 0.7)])
    right = _curve("right", [(0.0, 0.1, 0.7), (0.5, 0.2, 0.7), (1.0, 0.3, 0.7)])

    result = compare_melodies(left, right, window_size=0.5, step_size=0.5)

    assert result.level == "高度相似"
    assert result.modified_distance == pytest.approx(0.0)
    assert result.dtw_distance == pytest.approx(0.0)
    assert result.best_segment is not None
    assert result.score == pytest.approx(100.0)


def test_compare_melodies_detects_different_curves():
    left = _curve("left", [(0.0, 0.1, 0.7), (0.5, 0.2, 0.7), (1.0, 0.3, 0.7)])
    right = _curve("right", [(0.0, 0.8, 0.7), (0.5, 0.7, 0.7), (1.0, 0.6, 0.7)])

    result = compare_melodies(left, right, window_size=0.5, step_size=0.5)

    assert result.level == "不相似"
    assert result.score == 0.0


def test_classify_similarity_level_uses_segment_evidence():
    level = classify_similarity_level(0.30, 0.25, 0.03)

    assert level == "局部片段相似"


def test_similarity_score_validates_threshold():
    with pytest.raises(ValueError):
        similarity_score(0.1, suspicious_threshold=0.0)
