from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS
from src.io.midi_loader import load_midi
from src.models.melody_curve import MelodyCurve
from src.processing.normalization import normalize_minmax


DEFAULT_REFERENCE_ROOT = Path("data/midi_dataset_v1.1/data/raw")
DEFAULT_CACHE_PATH = Path("results/v1_1/standard_set_resampled64.npz")


@dataclass(frozen=True)
class ReferenceItem:
    """One labeled reference melody in the standard set."""

    name: str
    label: str
    filepath: str
    points: np.ndarray


@dataclass(frozen=True)
class NeighborMatch:
    """One nearest reference item used for KNN voting."""

    name: str
    label: str
    distance: float


@dataclass(frozen=True)
class QueryClassification:
    """KNN prediction for one query melody against the standard set."""

    query_name: str
    predicted_label: str
    neighbors: list[NeighborMatch]
    vote_counts: dict[str, int]
    mean_distances: dict[str, float]
    confidence: float


def resample_points(points: np.ndarray, target_points: int = 64) -> np.ndarray:
    """Resample normalized melody points to a fixed point count."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("points must be a 2D array")
    if target_points <= 0:
        raise ValueError("target_points must be positive")
    if len(array) == 0:
        return np.zeros((0, array.shape[1] if array.ndim == 2 else 3), dtype=np.float64)

    order = np.argsort(array[:, 0])
    array = array[order]
    unique_times, unique_indices = np.unique(array[:, 0], return_index=True)
    array = array[unique_indices]

    if len(array) == 1:
        sampled = np.repeat(array, target_points, axis=0)
        sampled[:, 0] = np.linspace(0.0, 1.0, target_points)
        return sampled

    grid = np.linspace(float(array[:, 0].min()), float(array[:, 0].max()), target_points)
    columns = [grid]
    for axis in range(1, array.shape[1]):
        columns.append(np.interp(grid, array[:, 0], array[:, axis]))
    return np.column_stack(columns)


def build_standard_set_cache(
    reference_root: Path | str = DEFAULT_REFERENCE_ROOT,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    target_points: int = 64,
) -> list[ReferenceItem]:
    """Build and save the standard-set cache from labeled genre directories."""
    root = Path(reference_root)
    if not root.is_dir():
        raise FileNotFoundError(f"reference root not found: {root}")

    curves = []
    for genre_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        genre = genre_dir.name
        midi_files = sorted(list(genre_dir.glob("*.mid")) + list(genre_dir.glob("*.midi")))
        for filepath in midi_files:
            curve = load_midi(str(filepath), extraction_mode="highest")
            if curve is None:
                continue
            curve.label = genre
            curves.append(curve)

    normalize_minmax(curves)
    items = [
        ReferenceItem(
            name=curve.name,
            label=str(curve.label),
            filepath=curve.filepath,
            points=resample_points(curve.points, target_points=target_points),
        )
        for curve in curves
        if curve.label and curve.points is not None and len(curve.points) > 0
    ]
    save_standard_set_cache(items, cache_path)
    return items


def save_standard_set_cache(items: list[ReferenceItem], cache_path: Path | str) -> None:
    """Save reference items to an NPZ cache."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        names=np.array([item.name for item in items]),
        labels=np.array([item.label for item in items]),
        filepaths=np.array([item.filepath for item in items]),
        points=np.array([item.points for item in items], dtype=np.float64),
    )


def load_standard_set_cache(cache_path: Path | str = DEFAULT_CACHE_PATH) -> list[ReferenceItem]:
    """Load reference items from an NPZ cache."""
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"standard-set cache not found: {path}")

    data = np.load(path)
    names = data["names"].astype(str).tolist()
    labels = data["labels"].astype(str).tolist()
    filepaths = data["filepaths"].astype(str).tolist()
    points = data["points"]
    return [
        ReferenceItem(name=name, label=label, filepath=filepath, points=point_array)
        for name, label, filepath, point_array in zip(names, labels, filepaths, points)
    ]


def load_or_build_standard_set(
    reference_root: Path | str = DEFAULT_REFERENCE_ROOT,
    cache_path: Path | str = DEFAULT_CACHE_PATH,
    target_points: int = 64,
) -> list[ReferenceItem]:
    """Load the standard-set cache, building it if needed."""
    path = Path(cache_path)
    if path.is_file():
        return load_standard_set_cache(path)
    return build_standard_set_cache(reference_root, path, target_points=target_points)


def load_query_curve(filepath: str, target_points: int = 64) -> MelodyCurve | None:
    """Load and resample one query MIDI using the standard-set preprocessing."""
    curve = load_midi(filepath, extraction_mode="highest")
    if curve is None:
        return None
    normalize_minmax([curve])
    if curve.points is not None:
        curve.points = resample_points(curve.points, target_points=target_points)
    return curve


def classify_query_curve(
    query_curve: MelodyCurve,
    references: list[ReferenceItem],
    method: str = "modified",
    k: int = 5,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> QueryClassification:
    """Classify one query curve against the standard set with KNN voting."""
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance method: {method}")
    if k <= 0:
        raise ValueError("k must be positive")
    if not references:
        raise ValueError("references must not be empty")
    if query_curve.points is None or len(query_curve.points) == 0:
        raise ValueError("query curve must have points")

    distance_function = DISTANCE_FUNCTIONS[method]
    query_points = _apply_weights(query_curve.points, weights)
    distances = []
    for item in references:
        reference_points = _apply_weights(item.points, weights)
        distances.append(float(distance_function(query_points, reference_points)))

    neighbor_count = min(k, len(references))
    ordered_indices = sorted(range(len(references)), key=lambda index: (distances[index], references[index].label))
    neighbor_indices = ordered_indices[:neighbor_count]
    neighbors = [
        NeighborMatch(
            name=references[index].name,
            label=references[index].label,
            distance=distances[index],
        )
        for index in neighbor_indices
    ]
    vote_counts = dict(Counter(neighbor.label for neighbor in neighbors))
    mean_distances = _mean_distances_by_label(neighbors)
    predicted_label = _vote_label(neighbors)
    confidence = vote_counts[predicted_label] / len(neighbors)
    return QueryClassification(
        query_curve.name,
        predicted_label,
        neighbors,
        vote_counts,
        mean_distances,
        confidence,
    )


def _vote_label(neighbors: list[NeighborMatch]) -> str:
    """Return majority-vote label, breaking ties by mean distance."""
    counts = {}
    distances = {}
    for neighbor in neighbors:
        counts[neighbor.label] = counts.get(neighbor.label, 0) + 1
        distances.setdefault(neighbor.label, []).append(neighbor.distance)
    best_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == best_count]
    return min(candidates, key=lambda label: (float(np.mean(distances[label])), label))


def _mean_distances_by_label(neighbors: list[NeighborMatch]) -> dict[str, float]:
    """Return mean neighbor distance grouped by label."""
    grouped = {}
    for neighbor in neighbors:
        grouped.setdefault(neighbor.label, []).append(neighbor.distance)
    return {
        label: float(np.mean(distances))
        for label, distances in grouped.items()
    }


def _apply_weights(points: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """Return a weighted copy of a point array."""
    weighted = np.array(points, dtype=np.float64, copy=True)
    for axis, weight in enumerate(weights):
        if axis < weighted.shape[1]:
            weighted[:, axis] *= weight
    return weighted
