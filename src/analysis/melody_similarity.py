from __future__ import annotations

from dataclasses import dataclass

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS
from src.analysis.reference_classifier import load_query_curve
from src.analysis.segment_analysis import SegmentMatch, find_best_segment_match
from src.models.melody_curve import MelodyCurve


@dataclass(frozen=True)
class MelodySimilarityResult:
    """Similarity and possible cover/borrowing evidence for two melodies."""

    left_name: str
    right_name: str
    modified_distance: float
    dtw_distance: float
    best_segment: SegmentMatch | None
    level: str
    score: float


def compare_melodies(
    left: MelodyCurve,
    right: MelodyCurve,
    high_threshold: float = 0.08,
    suspicious_threshold: float = 0.16,
    segment_threshold: float = 0.08,
    window_size: float = 0.25,
    step_size: float = 0.05,
) -> MelodySimilarityResult:
    """Compare two prepared melody curves and classify their similarity."""
    if left.points is None or right.points is None:
        raise ValueError("both melodies must have normalized points")
    if len(left.points) == 0 or len(right.points) == 0:
        raise ValueError("both melodies must contain points")

    modified_distance = float(DISTANCE_FUNCTIONS["modified"](left.points, right.points))
    dtw_distance = float(DISTANCE_FUNCTIONS["dtw"](left.points, right.points))
    best_segment = find_best_segment_match(
        left,
        right,
        method="modified",
        window_size=window_size,
        step_size=step_size,
        min_points=2,
    )
    level = classify_similarity_level(
        modified_distance,
        dtw_distance,
        best_segment.distance if best_segment is not None else None,
        high_threshold=high_threshold,
        suspicious_threshold=suspicious_threshold,
        segment_threshold=segment_threshold,
    )
    score = similarity_score(modified_distance, suspicious_threshold)
    return MelodySimilarityResult(
        left.name,
        right.name,
        modified_distance,
        dtw_distance,
        best_segment,
        level,
        score,
    )


def compare_midi_files(
    left_filepath: str,
    right_filepath: str,
    target_points: int = 64,
    high_threshold: float = 0.08,
    suspicious_threshold: float = 0.16,
    segment_threshold: float = 0.08,
    window_size: float = 0.25,
    step_size: float = 0.05,
) -> MelodySimilarityResult | None:
    """Load two MIDI files with standard preprocessing and compare them."""
    left = load_query_curve(left_filepath, target_points=target_points)
    right = load_query_curve(right_filepath, target_points=target_points)
    if left is None or right is None:
        return None
    return compare_melodies(
        left,
        right,
        high_threshold=high_threshold,
        suspicious_threshold=suspicious_threshold,
        segment_threshold=segment_threshold,
        window_size=window_size,
        step_size=step_size,
    )


def classify_similarity_level(
    modified_distance: float,
    dtw_distance: float,
    segment_distance: float | None,
    high_threshold: float = 0.08,
    suspicious_threshold: float = 0.16,
    segment_threshold: float = 0.08,
) -> str:
    """Return a compact Chinese label for the similarity evidence."""
    if modified_distance <= high_threshold and dtw_distance <= high_threshold:
        return "高度相似"
    if modified_distance <= suspicious_threshold:
        return "疑似翻唱/借鉴"
    if segment_distance is not None and segment_distance <= segment_threshold:
        return "局部片段相似"
    return "不相似"


def similarity_score(distance: float, suspicious_threshold: float = 0.16) -> float:
    """Map a distance to a 0-100 similarity score."""
    if suspicious_threshold <= 0:
        raise ValueError("suspicious_threshold must be positive")
    return max(0.0, min(100.0, 100.0 * (1.0 - distance / suspicious_threshold)))
