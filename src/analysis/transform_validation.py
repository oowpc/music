from __future__ import annotations

import copy
import random
from collections.abc import Callable

import numpy as np

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS
from src.models.melody_curve import MelodyCurve
from src.models.note import Note


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------


def apply_transposition(curve: MelodyCurve, semitones: int) -> MelodyCurve:
    """Shift all pitch values by *semitones* and clamp to [0, 127].

    Returns a **new** ``MelodyCurve`` with copied notes.  The original
    curve is never modified.
    """
    transposed_notes = []
    for raw_note in curve.raw_notes:
        new_pitch = min(max(raw_note.pitch + semitones, 0), 127)
        transposed_notes.append(
            Note(
                timestamp=raw_note.timestamp,
                pitch=new_pitch,
                velocity=raw_note.velocity,
            )
        )
    return MelodyCurve(
        name=curve.name,
        filepath=curve.filepath,
        label=curve.label,
        raw_notes=transposed_notes,
        color=curve.color,
    )


def apply_tempo_scale(curve: MelodyCurve, factor: float) -> MelodyCurve:
    """Multiply all timestamps by *factor*.

    Returns a **new** ``MelodyCurve`` with copied notes.  The original
    curve is never modified.
    """
    scaled_notes = []
    for raw_note in curve.raw_notes:
        scaled_notes.append(
            Note(
                timestamp=raw_note.timestamp * factor,
                pitch=raw_note.pitch,
                velocity=raw_note.velocity,
            )
        )
    return MelodyCurve(
        name=curve.name,
        filepath=curve.filepath,
        label=curve.label,
        raw_notes=scaled_notes,
        color=curve.color,
    )


def apply_rhythm_jitter(
    curve: MelodyCurve,
    std_seconds: float,
    seed: int = 42,
) -> MelodyCurve:
    """Add Gaussian noise N(0, *std_seconds*) to each timestamp.

    Notes are sorted by timestamp after jittering.  A fixed *seed* makes
    the output deterministic.

    Returns a **new** ``MelodyCurve`` with copied notes.  The original
    curve is never modified.
    """
    rng = np.random.default_rng(seed)
    jittered_notes = []
    for raw_note in curve.raw_notes:
        noise = float(rng.normal(loc=0.0, scale=std_seconds))
        jittered_notes.append(
            Note(
                timestamp=raw_note.timestamp + noise,
                pitch=raw_note.pitch,
                velocity=raw_note.velocity,
            )
        )
    jittered_notes.sort(key=lambda midi_note: midi_note.timestamp)
    return MelodyCurve(
        name=curve.name,
        filepath=curve.filepath,
        label=curve.label,
        raw_notes=jittered_notes,
        color=curve.color,
    )


# ---------------------------------------------------------------------------
# Experiment runner helpers
# ---------------------------------------------------------------------------


def _normalize_pair(
    original: MelodyCurve,
    transformed: MelodyCurve,
    normalize_fn: Callable[[list[MelodyCurve]], None],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize two curves together and return their point arrays.

    The curves are deep-copied first so the caller's originals are never
    mutated.  The normalizer (e.g. ``normalize_minmax``) receives a
    2-element list and is expected to set ``curve.points`` in-place.
    """
    left = copy.deepcopy(original)
    right = copy.deepcopy(transformed)
    normalize_fn([left, right])
    if left.points is None or right.points is None:
        raise RuntimeError("normalize_fn did not set curve.points")
    return left.points, right.points


# ---------------------------------------------------------------------------
# Experiment 1 — single-parameter transform sweep
# ---------------------------------------------------------------------------


def run_single_transform_test(
    curves: list[MelodyCurve],
    method: str,
    normalize_fn: Callable[[list[MelodyCurve]], None],
    transform_fn: Callable[[MelodyCurve, object], MelodyCurve],
    param_name: str,
    param_values: list[object],
) -> list[dict]:
    """Grid-search one transform parameter against a distance method.

    For every curve × parameter value combination the function:

    1. applies *transform_fn* with the given parameter value,
    2. normalises both the original and transformed curve together,
    3. computes the distance between the two point arrays with *method*.

    Returns a flat list of result dicts, each with keys ``curve_name``,
    ``param_name``, ``param_value``, ``distance``, and ``method``.
    """
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance method: {method}")

    distance_function = DISTANCE_FUNCTIONS[method]
    results: list[dict] = []

    for curve in curves:
        for param_value in param_values:
            transformed = transform_fn(curve, param_value)
            left_points, right_points = _normalize_pair(curve, transformed, normalize_fn)
            distance = float(distance_function(left_points, right_points))
            results.append(
                {
                    "curve_name": curve.name,
                    "param_name": param_name,
                    "param_value": param_value,
                    "distance": distance,
                    "method": method,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Experiment 2 — positive / negative pair separation
# ---------------------------------------------------------------------------


def run_pn_separation_test(
    curves: list[MelodyCurve],
    methods: list[str] | None = None,
    n_pairs: int = 10,
    normalize_fn: Callable[[list[MelodyCurve]], None] | None = None,
) -> dict:
    """Measure whether same-melody (positive) distances are smaller than
    different-melody (negative) distances.

    **Positive pairs** are a random curve and a +3-semitone transposition
    of itself.

    **Negative pairs** are two randomly chosen *different* curves.

    The function generates *n_pairs* of each type and evaluates every
    method in *methods* (default: all registered distance functions).

    Results are returned as a nested dict::

        {
            "<method>": {
                "positive_distances": [...],
                "negative_distances": [...],
            },
            ...
        }
    """
    selected_methods = methods or list(DISTANCE_FUNCTIONS.keys())
    for method in selected_methods:
        if method not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Unsupported distance method: {method}")

    if normalize_fn is None:
        raise ValueError("normalize_fn must be provided")

    rng = random.Random(42)
    curve_count = len(curves)
    if curve_count < 2:
        raise ValueError("at least two curves are required for P/N separation test")

    outcomes: dict = {method: {"positive_distances": [], "negative_distances": []} for method in selected_methods}

    for _ in range(n_pairs):
        # --- positive pair: random curve + its transposition (+3 semitones) ---
        pos_idx = rng.randrange(curve_count)
        pos_original = curves[pos_idx]
        pos_transposed = apply_transposition(pos_original, 3)
        pos_pair = _normalize_pair(pos_original, pos_transposed, normalize_fn)

        # --- negative pair: two different random curves ---
        neg_idx_a = rng.randrange(curve_count)
        neg_idx_b = rng.randrange(curve_count)
        while neg_idx_b == neg_idx_a:
            neg_idx_b = rng.randrange(curve_count)
        neg_a = curves[neg_idx_a]
        neg_b = curves[neg_idx_b]
        neg_pair = _normalize_pair(neg_a, neg_b, normalize_fn)

        for method in selected_methods:
            distance_function = DISTANCE_FUNCTIONS[method]
            outcomes[method]["positive_distances"].append(
                float(distance_function(pos_pair[0], pos_pair[1]))
            )
            outcomes[method]["negative_distances"].append(
                float(distance_function(neg_pair[0], neg_pair[1]))
            )

    return outcomes
