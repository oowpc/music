import numpy as np
import pytest

from src.analysis.segment_analysis import (
    find_best_segment_match,
    slice_points_by_time,
    sliding_windows,
)
from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.processing.normalization import normalize_minmax


def _curve(name, notes_data):
    """Create and normalize a test melody curve."""
    curve = MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[Note(timestamp, pitch, velocity) for timestamp, pitch, velocity in notes_data],
    )
    normalize_minmax([curve])
    return curve


def test_slice_points_by_time_returns_requested_range():
    points = np.array(
        [
            [0.0, 0.1, 0.5],
            [0.25, 0.2, 0.5],
            [0.5, 0.3, 0.5],
            [0.75, 0.4, 0.5],
        ]
    )

    sliced = slice_points_by_time(points, 0.2, 0.5)

    np.testing.assert_allclose(sliced[:, 0], [0.25, 0.5])


def test_sliding_windows_skips_sparse_windows():
    points = np.array(
        [
            [0.0, 0.1, 0.5],
            [0.2, 0.2, 0.5],
            [0.4, 0.3, 0.5],
            [0.8, 0.4, 0.5],
        ]
    )

    windows = sliding_windows(points, window_size=0.25, step_size=0.2, min_points=2)

    assert [(round(start, 2), round(end, 2), len(segment)) for start, end, segment in windows] == [
        (0.0, 0.25, 2),
        (0.2, 0.45, 2),
    ]


def test_find_best_segment_match_finds_shared_motif():
    left = _curve(
        "left",
        [
            (0.0, 60, 80),
            (1.0, 62, 80),
            (2.0, 64, 80),
            (3.0, 70, 80),
            (4.0, 72, 80),
        ],
    )
    right = _curve(
        "right",
        [
            (0.0, 50, 80),
            (1.0, 52, 80),
            (2.0, 60, 80),
            (3.0, 62, 80),
            (4.0, 64, 80),
        ],
    )

    match = find_best_segment_match(
        left,
        right,
        method="modified",
        window_size=0.5,
        step_size=0.5,
        min_points=2,
    )

    assert match is not None
    assert match.distance < 0.05
    assert match.left_start == pytest.approx(0.0)
    assert match.right_start == pytest.approx(0.5)


def test_find_best_segment_match_returns_none_without_points():
    left = MelodyCurve(name="left", filepath="/fake/left.mid")
    right = MelodyCurve(name="right", filepath="/fake/right.mid")

    assert find_best_segment_match(left, right) is None


def test_find_best_segment_match_rejects_unknown_method():
    left = _curve("left", [(0.0, 60, 80), (1.0, 62, 80)])
    right = _curve("right", [(0.0, 60, 80), (1.0, 62, 80)])

    with pytest.raises(ValueError):
        find_best_segment_match(left, right, method="unknown")
