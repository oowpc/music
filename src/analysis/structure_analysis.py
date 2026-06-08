from __future__ import annotations

from dataclasses import dataclass
import string

import numpy as np

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS
from src.analysis.segment_analysis import slice_points_by_time
from src.models.melody_curve import MelodyCurve


@dataclass(frozen=True)
class StructureSegment:
    """One fixed-time section used for macro-structure analysis."""

    index: int
    start: float
    end: float
    points: int
    label: str
    base_label: str


@dataclass(frozen=True)
class StructureRelation:
    """A repeated or varied relationship between two sections."""

    left_index: int
    right_index: int
    distance: float
    relation: str


@dataclass(frozen=True)
class StructureAnalysisResult:
    """Detected section labels and macro structure for one melody curve."""

    curve_name: str
    segments: list[StructureSegment]
    relations: list[StructureRelation]
    section_sequence: str
    base_sequence: str
    macro_structure: str
    summary: str


def analyze_structure(
    curve: MelodyCurve,
    section_count: int = 8,
    method: str = "modified",
    repeat_threshold: float = 0.08,
    variation_threshold: float = 0.2,
    min_points: int = 2,
) -> StructureAnalysisResult | None:
    """Detect repeated sections, variations, and ABA-like macro structure.

    The melody is split into equal normalized-time sections. Each section is
    compared with prior sections after localizing time to the section itself, so
    absolute position does not dominate the shape comparison.
    """
    if section_count < 2:
        raise ValueError("section_count must be at least 2")
    if repeat_threshold < 0:
        raise ValueError("repeat_threshold must be non-negative")
    if variation_threshold < repeat_threshold:
        raise ValueError("variation_threshold must be greater than or equal to repeat_threshold")
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance method: {method}")
    if curve.points is None:
        return None

    windows = _fixed_sections(curve.points, section_count, min_points=min_points)
    if len(windows) < 2:
        return None

    distance_function = DISTANCE_FUNCTIONS[method]
    segments: list[StructureSegment] = []
    relations: list[StructureRelation] = []
    variation_counts: dict[str, int] = {}

    for index, (start, end, points) in enumerate(windows):
        if index == 0:
            base_label = _label_for_index(0)
            label = base_label
        else:
            nearest_index, nearest_distance = _nearest_previous_segment(points, windows, index, distance_function)
            nearest_segment = segments[nearest_index]
            if nearest_distance <= repeat_threshold:
                base_label = nearest_segment.base_label
                label = nearest_segment.label
                relations.append(StructureRelation(nearest_index, index, nearest_distance, "重复"))
            elif nearest_distance <= variation_threshold:
                base_label = nearest_segment.base_label
                variation_counts[base_label] = variation_counts.get(base_label, 0) + 1
                label = f"{base_label}{variation_counts[base_label]}'"
                relations.append(StructureRelation(nearest_index, index, nearest_distance, "变奏"))
            else:
                base_label = _label_for_index(len({segment.base_label for segment in segments}))
                label = base_label

        segments.append(
            StructureSegment(
                index=index,
                start=start,
                end=end,
                points=len(points),
                label=label,
                base_label=base_label,
            )
        )

    section_sequence = "-".join(segment.label for segment in segments)
    base_sequence = "-".join(segment.base_label for segment in segments)
    macro_structure = _detect_macro_structure([segment.base_label for segment in segments])
    summary = _build_summary(segments, relations, macro_structure)
    return StructureAnalysisResult(
        curve_name=curve.name,
        segments=segments,
        relations=relations,
        section_sequence=section_sequence,
        base_sequence=base_sequence,
        macro_structure=macro_structure,
        summary=summary,
    )


def analyze_structures(
    curves: list[MelodyCurve],
    section_count: int = 8,
    method: str = "modified",
    repeat_threshold: float = 0.08,
    variation_threshold: float = 0.2,
    min_points: int = 2,
) -> list[StructureAnalysisResult]:
    """Analyze all curves that have enough normalized points."""
    results = []
    for curve in curves:
        result = analyze_structure(
            curve,
            section_count=section_count,
            method=method,
            repeat_threshold=repeat_threshold,
            variation_threshold=variation_threshold,
            min_points=min_points,
        )
        if result is not None:
            results.append(result)
    return results


def _fixed_sections(
    points: np.ndarray,
    section_count: int,
    min_points: int = 2,
) -> list[tuple[float, float, np.ndarray]]:
    """Split points into fixed normalized-time sections."""
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return []
    if array.ndim != 2:
        raise ValueError("points must be a 2D array")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    if len(array) == 0:
        return []

    start_time = float(np.min(array[:, 0]))
    end_time = float(np.max(array[:, 0]))
    duration = end_time - start_time
    if duration <= 0:
        return []

    sections = []
    for index in range(section_count):
        start = start_time + duration * index / section_count
        end = start_time + duration * (index + 1) / section_count
        if index == section_count - 1:
            end += 1e-12
        segment = _localize_section(slice_points_by_time(array, start, end), start, end)
        if len(segment) >= min_points:
            sections.append((start, min(end, end_time), segment))
    return sections


def _localize_section(points: np.ndarray, start: float, end: float) -> np.ndarray:
    """Normalize a section's own time span to [0, 1]."""
    localized = np.array(points, dtype=np.float64, copy=True)
    if len(localized) == 0:
        return localized
    segment_start = float(np.min(localized[:, 0]))
    segment_end = float(np.max(localized[:, 0]))
    duration = segment_end - segment_start
    localized[:, 0] = (localized[:, 0] - segment_start) / duration if duration > 0 else 0.0
    return localized


def _nearest_previous_segment(points, windows, current_index, distance_function) -> tuple[int, float]:
    """Return the nearest previous section index and distance."""
    best_index = 0
    best_distance = float("inf")
    for candidate_index in range(current_index):
        distance = float(distance_function(points, windows[candidate_index][2]))
        if distance < best_distance:
            best_index = candidate_index
            best_distance = distance
    return best_index, best_distance


def _detect_macro_structure(base_labels: list[str]) -> str:
    """Detect simple macro forms from base section labels."""
    collapsed = []
    for label in base_labels:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)

    if len(collapsed) >= 3:
        for index in range(len(collapsed) - 2):
            left, middle, right = collapsed[index : index + 3]
            if left == right and left != middle:
                return "ABA"
    if any(base_labels.count(label) > 1 for label in set(base_labels)):
        return "重复型"
    return "通谱型"


def _build_summary(
    segments: list[StructureSegment],
    relations: list[StructureRelation],
    macro_structure: str,
) -> str:
    """Build a compact Chinese summary for UI display."""
    repeat_count = sum(1 for relation in relations if relation.relation == "重复")
    variation_count = sum(1 for relation in relations if relation.relation == "变奏")
    return (
        f"{len(segments)} 段，结构 {macro_structure}，"
        f"重复 {repeat_count} 处，变奏 {variation_count} 处"
    )


def _label_for_index(index: int) -> str:
    """Return A, B, ... Z, AA, AB labels for arbitrary section counts."""
    alphabet = string.ascii_uppercase
    label = ""
    value = index
    while True:
        label = alphabet[value % len(alphabet)] + label
        value = value // len(alphabet) - 1
        if value < 0:
            return label
