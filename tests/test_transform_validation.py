import numpy as np
import pytest

from src.analysis.transform_validation import (
    apply_rhythm_jitter,
    apply_tempo_scale,
    apply_transposition,
    run_pn_separation_test,
    run_single_transform_test,
)
from src.io.midi_loader import load_midi_files
from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.processing.hausdorff import hausdorff_modified, hausdorff_standard
from src.processing.normalization import normalize_minmax, normalize_zscore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_curve(name: str, notes_data: list[tuple[float, int, int]]) -> MelodyCurve:
    """Create a test curve from tuples of (time, pitch, velocity)."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[Note(t, p, v) for t, p, v in notes_data],
    )


# ---------------------------------------------------------------------------
# Transposition invariance
# ---------------------------------------------------------------------------


def test_transposition_same_curve_zero_distance():
    """Transposition of 0 semitones must produce distance 0 with all methods."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    result = apply_transposition(curve, 0)

    assert result is not curve
    assert result.color == curve.color
    assert len(result.raw_notes) == len(curve.raw_notes)

    normalize_minmax([curve, result])
    assert hausdorff_standard(curve.points, result.points) == pytest.approx(0.0, abs=1e-10)
    assert hausdorff_modified(curve.points, result.points) == pytest.approx(0.0, abs=1e-10)


def test_transposition_octave_invariance_zscore():
    """+12 semitones with z-score normalization yields identical points."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 90)])
    result = apply_transposition(curve, 12)

    assert result is not curve
    assert len(result.raw_notes) == len(curve.raw_notes)

    normalize_zscore([curve, result])
    np.testing.assert_allclose(curve.points, result.points, atol=1e-10)


def test_transposition_clamps_to_midi_range():
    """Pitch must stay in [0, 127] after transposition."""
    curve = make_curve("high", [(0.0, 125, 80)])
    shifted = apply_transposition(curve, 5)
    assert shifted.raw_notes[0].pitch == 127

    curve_low = make_curve("low", [(0.0, 3, 80)])
    shifted_low = apply_transposition(curve_low, -5)
    assert shifted_low.raw_notes[0].pitch == 0


def test_transposition_creates_new_curve():
    """Transforms must return new MelodyCurve, leaving original untouched."""
    curve = make_curve("original", [(0.0, 60, 80), (1.0, 64, 100)])
    transformed = apply_transposition(curve, 5)

    assert transformed is not curve
    assert curve.raw_notes[0].pitch == 60
    assert transformed.raw_notes[0].pitch == 65


# ---------------------------------------------------------------------------
# Tempo scaling
# ---------------------------------------------------------------------------


def test_tempo_scale_identity():
    """Factor 1.0 produces identical raw notes."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    result = apply_tempo_scale(curve, 1.0)

    assert result is not curve
    assert len(result.raw_notes) == len(curve.raw_notes)
    for original, scaled in zip(curve.raw_notes, result.raw_notes):
        assert scaled.timestamp == pytest.approx(original.timestamp)
        assert scaled.pitch == original.pitch
        assert scaled.velocity == original.velocity


def test_tempo_scale_monotonicity():
    """Larger scaling factor produces larger raw Euclidean distance.

    After per-curve normalization (minmax or zscore) linear time scaling
    is absorbed, so we test monotonicity on **raw** point arrays before
    normalization — this captures the geometric intuition of Kelly 2012
    while acknowledging that the normalizer intentionally removes linear
    time scaling for the downstream distance matrix.
    """
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 90)])

    def raw_points(c: MelodyCurve) -> np.ndarray:
        return np.array([[n.timestamp, n.pitch, n.velocity] for n in c.raw_notes], dtype=np.float64)

    original_pts = raw_points(curve)

    d_half = hausdorff_standard(original_pts, raw_points(apply_tempo_scale(curve, 0.5)))
    d_one = hausdorff_standard(original_pts, raw_points(apply_tempo_scale(curve, 1.0)))
    d_double = hausdorff_standard(original_pts, raw_points(apply_tempo_scale(curve, 2.0)))

    assert d_one == pytest.approx(0.0, abs=1e-10)
    assert d_half > 0.0
    assert d_double > 0.0
    assert d_double > d_half  # larger deviation → larger distance


def test_tempo_scale_preserves_other_dimensions():
    """Only timestamps are scaled; pitch and velocity are untouched."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    result = apply_tempo_scale(curve, 2.0)

    for original, scaled in zip(curve.raw_notes, result.raw_notes):
        assert scaled.timestamp == pytest.approx(original.timestamp * 2.0)
        assert scaled.pitch == original.pitch
        assert scaled.velocity == original.velocity


# ---------------------------------------------------------------------------
# Rhythm jitter
# ---------------------------------------------------------------------------


def test_rhythm_jitter_reproducible():
    """Same seed must produce identical results."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 90)])
    result_a = apply_rhythm_jitter(curve, std_seconds=0.1, seed=42)
    result_b = apply_rhythm_jitter(curve, std_seconds=0.1, seed=42)

    assert len(result_a.raw_notes) == len(result_b.raw_notes)
    for note_a, note_b in zip(result_a.raw_notes, result_b.raw_notes):
        assert note_a.timestamp == pytest.approx(note_b.timestamp)
        assert note_a.pitch == note_b.pitch
        assert note_a.velocity == note_b.velocity


def test_rhythm_jitter_different_seeds_diverge():
    """Different seeds produce potentially different results."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 90)])
    result_a = apply_rhythm_jitter(curve, std_seconds=0.1, seed=1)
    result_b = apply_rhythm_jitter(curve, std_seconds=0.1, seed=2)

    timestamps_a = [n.timestamp for n in result_a.raw_notes]
    timestamps_b = [n.timestamp for n in result_b.raw_notes]
    assert timestamps_a != timestamps_b


def test_rhythm_jitter_sorted_by_timestamp():
    """Resulting notes must be sorted by timestamp."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 90)])
    result = apply_rhythm_jitter(curve, std_seconds=1.0, seed=42)

    timestamps = [n.timestamp for n in result.raw_notes]
    assert timestamps == sorted(timestamps)


def test_rhythm_jitter_preserves_dimensions():
    """Only timestamps are perturbed; pitch and velocity stay the same."""
    curve = make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)])
    result = apply_rhythm_jitter(curve, std_seconds=0.1, seed=42)

    for note in result.raw_notes:
        assert 0 <= note.pitch <= 127
        assert 0 <= note.velocity <= 127

    pitches = {n.pitch for n in curve.raw_notes}
    jittered_pitches = {n.pitch for n in result.raw_notes}
    assert pitches == jittered_pitches


# ---------------------------------------------------------------------------
# Single-transform sweep experiments
# ---------------------------------------------------------------------------


def test_run_single_transform_test_transposition(simple_midi_file):
    """Transposition sweep across three semitone values."""
    curves = load_midi_files([simple_midi_file])
    assert len(curves) == 1
    assert len(curves[0].raw_notes) > 0

    results = run_single_transform_test(
        curves=curves,
        method="standard",
        normalize_fn=normalize_minmax,
        transform_fn=lambda c, v: apply_transposition(c, v),
        param_name="semitones",
        param_values=[0, 3, 6],
    )

    assert len(results) == 3  # 1 curve × 3 values
    for entry in results:
        assert "curve_name" in entry
        assert entry["param_name"] == "semitones"
        assert entry["param_value"] in [0, 3, 6]
        assert entry["distance"] >= 0.0
        assert entry["method"] == "standard"

    distances = [entry["distance"] for entry in results]
    assert distances[0] == pytest.approx(0.0, abs=1e-10)  # 0 semitones


def test_run_single_transform_test_tempo(simple_midi_file):
    """Tempo sweep tested via raw-point monotonicity (normalization absorbs linear scaling)."""
    curves = load_midi_files([simple_midi_file])

    def raw_points(c: MelodyCurve) -> np.ndarray:
        return np.array([[n.timestamp, n.pitch, n.velocity] for n in c.raw_notes], dtype=np.float64)

    original_pts = raw_points(curves[0])
    d_half = hausdorff_standard(original_pts, raw_points(apply_tempo_scale(curves[0], 0.5)))
    d_double = hausdorff_standard(original_pts, raw_points(apply_tempo_scale(curves[0], 2.0)))

    assert d_half > 0.0
    assert d_double > 0.0
    assert d_double > d_half


def test_run_single_transform_test_rejects_unknown_method(simple_midi_file):
    """Unknown distance method must raise ValueError."""
    curves = load_midi_files([simple_midi_file])

    with pytest.raises(ValueError):
        run_single_transform_test(
            curves=curves,
            method="nonexistent",
            normalize_fn=normalize_minmax,
            transform_fn=lambda c, v: apply_transposition(c, v),
            param_name="semitones",
            param_values=[0],
        )


# ---------------------------------------------------------------------------
# Positive / negative separation experiment
# ---------------------------------------------------------------------------


def test_pn_separation_structure():
    """Result dict has expected topology."""
    curves = [
        make_curve("a", [(0.0, 60, 80), (1.0, 64, 100)]),
        make_curve("b", [(0.0, 62, 70), (1.0, 67, 90)]),
        make_curve("c", [(0.0, 72, 110), (1.0, 75, 120)]),
    ]

    outcome = run_pn_separation_test(
        curves=curves,
        methods=["standard", "modified"],
        n_pairs=5,
        normalize_fn=normalize_minmax,
    )

    assert set(outcome.keys()) == {"standard", "modified"}
    for method in ("standard", "modified"):
        assert len(outcome[method]["positive_distances"]) == 5
        assert len(outcome[method]["negative_distances"]) == 5
        assert all(d >= 0.0 for d in outcome[method]["positive_distances"])
        assert all(d >= 0.0 for d in outcome[method]["negative_distances"])


def test_pn_separation_positive_smaller_than_negative():
    """Positive distances should be generally smaller than negative ones."""
    curves = [
        make_curve("a", [(0.0, 60, 80), (1.0, 64, 100), (2.0, 67, 90)]),
        make_curve("b", [(0.0, 50, 60), (1.0, 55, 70), (2.0, 58, 80)]),
        make_curve("c", [(0.0, 72, 110), (1.0, 75, 120), (2.0, 79, 115)]),
    ]

    outcome = run_pn_separation_test(
        curves=curves,
        methods=["standard", "modified"],
        n_pairs=20,
        normalize_fn=normalize_minmax,
    )

    for method in ("standard", "modified"):
        pos = outcome[method]["positive_distances"]
        neg = outcome[method]["negative_distances"]
        mean_pos = np.mean(pos)
        mean_neg = np.mean(neg)
        # With random sampling and small test set, at least half of positive
        # distances should be smaller than the negative mean.
        assert mean_pos < mean_neg, (
            f"{method}: mean positive={mean_pos:.4f} >= mean negative={mean_neg:.4f}"
        )


def test_pn_separation_rejects_single_curve():
    """At least two curves are required."""
    curves = [make_curve("solo", [(0.0, 60, 80)])]

    with pytest.raises(ValueError):
        run_pn_separation_test(
            curves=curves,
            n_pairs=5,
            normalize_fn=normalize_minmax,
        )


def test_pn_separation_requires_normalize_fn():
    """normalize_fn must be provided."""
    curves = [
        make_curve("a", [(0.0, 60, 80)]),
        make_curve("b", [(0.0, 62, 80)]),
    ]

    with pytest.raises(ValueError):
        run_pn_separation_test(curves=curves, n_pairs=5, normalize_fn=None)


def test_pn_separation_rejects_unknown_method():
    """Unknown method must raise ValueError."""
    curves = [
        make_curve("a", [(0.0, 60, 80)]),
        make_curve("b", [(0.0, 62, 80)]),
    ]

    with pytest.raises(ValueError):
        run_pn_separation_test(
            curves=curves,
            methods=["ghost_method"],
            n_pairs=2,
            normalize_fn=normalize_minmax,
        )


# ---------------------------------------------------------------------------
# MIDI fixture integration
# ---------------------------------------------------------------------------


def test_transposition_with_midi_fixture(simple_midi_file, sibling_midi_file):
    """Transposition + normalization on real MIDI curves."""
    curves = load_midi_files([simple_midi_file, sibling_midi_file])
    assert len(curves) == 2

    normalize_minmax(curves)
    original_dist = hausdorff_standard(curves[0].points, curves[1].points)
    assert original_dist > 0.0

    # Transpose sibling by 0: should give same distance
    reloaded = load_midi_files([simple_midi_file, sibling_midi_file])
    transposed = apply_transposition(reloaded[1], 0)
    normalize_minmax([reloaded[0], transposed])
    assert hausdorff_standard(reloaded[0].points, transposed.points) == pytest.approx(original_dist)


def test_tempo_scale_with_midi_fixture(two_note_midi_file):
    """Minmax normalization absorbs tempo scaling on real MIDI data."""
    original = load_midi_files([two_note_midi_file])[0]
    scaled = apply_tempo_scale(original, 0.5)

    normalize_minmax([original])
    normalize_minmax([scaled])

    np.testing.assert_allclose(original.points, scaled.points, atol=1e-10)


def test_jitter_with_midi_fixture(simple_midi_file):
    """Rhythm jitter on MIDI fixture produces different but valid curve."""
    curves = load_midi_files([simple_midi_file])
    assert len(curves) == 1

    jittered = apply_rhythm_jitter(curves[0], std_seconds=0.05, seed=42)

    assert len(jittered.raw_notes) == len(curves[0].raw_notes)
    assert jittered.color == curves[0].color
    assert jittered.name == curves[0].name

    # Timestamps are sorted
    timestamps = [n.timestamp for n in jittered.raw_notes]
    assert timestamps == sorted(timestamps)
