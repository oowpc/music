import numpy as np
import pytest

from src.analysis.retrieval import evaluate_retrieval, top_k_retrieve
from src.models.melody_curve import MelodyCurve


def _make_curve(
    name: str,
    label: str | None = None,
    points: np.ndarray | None = None,
) -> MelodyCurve:
    """Create a labeled MelodyCurve with optional normalized points."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        label=label,
        points=points,
    )


def _ascending_points(start: float = 0.0, count: int = 5) -> np.ndarray:
    """Return (count, 3) points with ascending pitch values."""
    return np.array(
        [(start + i * 0.2, i / 10.0, 0.5) for i in range(count)],
        dtype=np.float64,
    )


class TestTopKRetrieve:
    def test_self_first(self):
        """Query with k=1 on a DB containing itself → first result is self."""
        query = _make_curve("query", label="pop", points=_ascending_points(0.0, 5))
        other = _make_curve("other", label="rock", points=_ascending_points(0.5, 5))
        database = [other, query, other]  # query is at index 1

        results = top_k_retrieve(query, database, k=1)

        assert len(results) == 1
        assert results[0]["index"] == 1
        assert results[0]["name"] == "query"
        assert results[0]["distance"] == 0.0

    def test_returns_k_results(self):
        """With k=3 and 10 items in DB → returns 3 results."""
        query = _make_curve("query", points=_ascending_points(0.0, 5))
        database = [
            _make_curve(f"curve_{i}", points=_ascending_points(i * 0.1, 5))
            for i in range(10)
        ]

        results = top_k_retrieve(query, database, k=3)

        assert len(results) == 3
        for entry in results:
            assert "index" in entry
            assert "name" in entry
            assert "label" in entry
            assert "distance" in entry

    def test_skips_none_points(self):
        """Database entries with None points are skipped."""
        query = _make_curve("query", points=_ascending_points(0.0, 5))
        database = [
            _make_curve("no_points", points=None),
            _make_curve("valid", points=_ascending_points(0.3, 5)),
        ]

        results = top_k_retrieve(query, database, k=3)

        assert len(results) == 1
        assert results[0]["name"] == "valid"

    def test_rejects_invalid_method(self):
        """Unknown method raises ValueError."""
        query = _make_curve("query", points=_ascending_points(0.0, 5))
        with pytest.raises(ValueError, match="Unsupported distance method"):
            top_k_retrieve(query, [query], method="nonexistent", k=1)

    def test_rejects_zero_k(self):
        """Zero k raises ValueError."""
        query = _make_curve("query", points=_ascending_points(0.0, 5))
        with pytest.raises(ValueError, match="k must be positive"):
            top_k_retrieve(query, [query], k=0)

    def test_rejects_query_without_points(self):
        """Query without points raises ValueError."""
        query = _make_curve("query", points=None)
        with pytest.raises(ValueError, match="query.points must not be None"):
            top_k_retrieve(query, [query], k=1)


class TestEvaluateRetrieval:
    def test_perfect_retrieval(self):
        """Two curves with same label → precision_at_k should be 1.0."""
        curves = [
            _make_curve("a", label="pop", points=_ascending_points(0.0, 5)),
            _make_curve("b", label="pop", points=_ascending_points(0.05, 5)),
        ]

        result = evaluate_retrieval(curves, k=1)

        assert result["total"] == 2
        assert result["hit_count"] == 2
        assert result["precision_at_k"] == pytest.approx(1.0)
        assert result["per_label_precision"]["pop"] == pytest.approx(1.0)

    def test_no_labels(self):
        """Curves without labels → evaluation returns empty dict."""
        curves = [
            _make_curve("a", label=None, points=_ascending_points(0.0, 5)),
            _make_curve("b", label=None, points=_ascending_points(0.1, 5)),
        ]

        result = evaluate_retrieval(curves, k=1)

        assert result == {}

    def test_mixed_labels_only_uses_labeled(self):
        """Only labeled curves participate in evaluation."""
        curves = [
            _make_curve("a", label="pop", points=_ascending_points(0.0, 5)),
            _make_curve("b", label="pop", points=_ascending_points(0.05, 5)),
            _make_curve("c", label=None, points=_ascending_points(0.1, 5)),
            _make_curve("d", label=None, points=_ascending_points(0.15, 5)),
        ]

        result = evaluate_retrieval(curves, k=1)

        assert result["total"] == 2  # only labeled curves count
        assert result["precision_at_k"] == pytest.approx(1.0)

    def test_cross_label_separation(self):
        """Curves with different labels should not retrieve each other."""
        curves = [
            _make_curve("a", label="pop", points=_ascending_points(0.0, 5)),
            _make_curve("b", label="rock", points=_ascending_points(0.8, 5)),
            _make_curve("c", label="pop", points=_ascending_points(0.05, 5)),
            _make_curve("d", label="rock", points=_ascending_points(0.75, 5)),
        ]

        result = evaluate_retrieval(curves, k=1)

        assert result["total"] == 4
        assert result["hit_count"] == 4
        assert result["precision_at_k"] == pytest.approx(1.0)
