import pytest

from src.analysis.evaluation import evaluate
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
