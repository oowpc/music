import numpy as np

from src.analysis.structure_analysis import analyze_structure, analyze_structures
from src.models.melody_curve import MelodyCurve


def _curve(name: str, points: list[tuple[float, float, float]]) -> MelodyCurve:
    """Create a normalized curve for structure-analysis tests."""
    return MelodyCurve(name=name, filepath=f"/fake/{name}.mid", points=np.array(points, dtype=float))


def test_analyze_structure_detects_repetition_and_aba():
    curve = _curve(
        "aba",
        [
            (0.00, 0.40, 0.7),
            (0.15, 0.50, 0.7),
            (0.30, 0.60, 0.7),
            (0.36, 0.80, 0.7),
            (0.50, 0.70, 0.7),
            (0.63, 0.85, 0.7),
            (0.70, 0.40, 0.7),
            (0.85, 0.50, 0.7),
            (1.00, 0.60, 0.7),
        ],
    )

    result = analyze_structure(curve, section_count=3, repeat_threshold=0.03, variation_threshold=0.12)

    assert result is not None
    assert result.section_sequence == "A-B-A"
    assert result.base_sequence == "A-B-A"
    assert result.macro_structure == "ABA"
    assert [relation.relation for relation in result.relations] == ["重复"]


def test_analyze_structure_marks_variations_with_base_label():
    curve = _curve(
        "variation",
        [
            (0.00, 0.40, 0.7),
            (0.15, 0.50, 0.7),
            (0.30, 0.60, 0.7),
            (0.36, 0.44, 0.7),
            (0.50, 0.54, 0.7),
            (0.63, 0.64, 0.7),
            (0.70, 0.80, 0.7),
            (0.85, 0.90, 0.7),
            (1.00, 0.75, 0.7),
        ],
    )

    result = analyze_structure(curve, section_count=3, repeat_threshold=0.02, variation_threshold=0.08)

    assert result is not None
    assert result.section_sequence == "A-A1'-B"
    assert result.base_sequence == "A-A-B"
    assert result.relations[0].relation == "变奏"


def test_analyze_structures_skips_curves_without_enough_points():
    enough = _curve(
        "enough",
        [
            (0.0, 0.1, 0.7),
            (0.2, 0.2, 0.7),
            (0.8, 0.1, 0.7),
            (1.0, 0.2, 0.7),
        ],
    )
    empty = _curve("empty", [])

    results = analyze_structures([enough, empty], section_count=2, min_points=2)

    assert [result.curve_name for result in results] == ["enough"]
