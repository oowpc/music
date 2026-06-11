from __future__ import annotations

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS
from src.models.melody_curve import MelodyCurve


def top_k_retrieve(
    query: MelodyCurve,
    database: list[MelodyCurve],
    method: str = "modified",
    k: int = 5,
) -> list[dict]:
    """Return the *k* nearest database curves to a *query* curve.

    Parameters
    ----------
    query:
        The melody curve whose neighbors are requested.
    database:
        Curves to search.  Entries with ``points=None`` are skipped.
    method:
        Key into :data:`src.analysis.distance_matrix.DISTANCE_FUNCTIONS`.
    k:
        Maximum number of results (fewer are returned when the database
        is smaller).

    Returns
    -------
    list[dict]
        Sorted by ascending distance.  Each dict has keys ``index``
        (position in the original *database* list), ``name``, ``label``,
        and ``distance``.  When *query* appears in *database* by identity
        it is always placed first with ``distance == 0``.
    """
    if method not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance method: {method}")
    if k <= 0:
        raise ValueError("k must be positive")
    if query.points is None:
        raise ValueError("query.points must not be None")

    distance_fn = DISTANCE_FUNCTIONS[method]
    entries: list[dict] = []

    for index, db_curve in enumerate(database):
        if db_curve.points is None:
            continue
        is_self = query is db_curve
        if is_self:
            distance = 0.0
        else:
            distance = float(distance_fn(query.points, db_curve.points))
        entries.append(
            {
                "index": index,
                "name": db_curve.name,
                "label": db_curve.label,
                "distance": distance,
                "_self": is_self,
            }
        )

    entries.sort(key=lambda entry: (entry["distance"], 0 if entry["_self"] else 1))
    top_k = entries[:k]
    for entry in top_k:
        del entry["_self"]
    return top_k


def evaluate_retrieval(
    curves: list[MelodyCurve],
    method: str = "modified",
    k: int = 5,
) -> dict:
    """Leave-one-out retrieval evaluation on labeled curves.

    For every curve that carries a label, the function treats it as a
    query and searches the remaining labeled curves.  A query is counted
    as a hit when the top-*k* results contain at least one curve that
    shares the same label.

    Curves without labels are ignored entirely.

    Returns
    -------
    dict
        ``precision_at_k``, ``hit_count``, ``total``, and
        ``per_label_precision``.  Returns an empty dict when no curve
        has a label.
    """
    labeled = [curve for curve in curves if curve.label]
    if not labeled:
        return {}

    hit_count = 0
    per_label: dict[str, dict[str, int]] = {}

    for query in labeled:
        database = [curve for curve in labeled if curve is not query]
        results = top_k_retrieve(query, database, method=method, k=k)
        query_label = str(query.label)
        hit = any(result["label"] == query_label for result in results)

        per_label.setdefault(query_label, {"hits": 0, "total": 0})
        per_label[query_label]["total"] += 1
        if hit:
            hit_count += 1
            per_label[query_label]["hits"] += 1

    total = len(labeled)
    per_label_precision = {
        label: data["hits"] / data["total"] for label, data in per_label.items()
    }

    return {
        "precision_at_k": hit_count / total if total else 0.0,
        "hit_count": hit_count,
        "total": total,
        "per_label_precision": per_label_precision,
    }
