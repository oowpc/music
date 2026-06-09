from collections import Counter, defaultdict

import numpy as np

from src.models.melody_curve import MelodyCurve


def _combinations_2(count: int) -> int:
    """Return count choose 2 for non-negative integer counts."""
    return count * (count - 1) // 2


def _adjusted_rand_score(true_labels: list[str], cluster_labels: list[int]) -> float:
    """Return adjusted Rand index for equal-length true and cluster labels."""
    sample_count = len(true_labels)
    if sample_count < 2:
        return 1.0

    contingency: defaultdict[tuple[str, int], int] = defaultdict(int)
    true_counts: Counter[str] = Counter()
    cluster_counts: Counter[int] = Counter()

    for true_label, cluster_label in zip(true_labels, cluster_labels):
        contingency[(true_label, cluster_label)] += 1
        true_counts[true_label] += 1
        cluster_counts[cluster_label] += 1

    index = sum(_combinations_2(count) for count in contingency.values())
    true_sum = sum(_combinations_2(count) for count in true_counts.values())
    cluster_sum = sum(_combinations_2(count) for count in cluster_counts.values())
    total_pairs = _combinations_2(sample_count)

    expected = true_sum * cluster_sum / total_pairs if total_pairs else 0.0
    maximum = (true_sum + cluster_sum) / 2.0
    denominator = maximum - expected
    if denominator == 0:
        return 1.0 if index == maximum else 0.0
    return (index - expected) / denominator


def _compute_purity(true_labels: list[str], cluster_labels: list[int]) -> float:
    """Return cluster purity for equal-length true and predicted labels."""
    clusters: defaultdict[int, list[str]] = defaultdict(list)
    for true_label, cluster_label in zip(true_labels, cluster_labels):
        clusters[cluster_label].append(true_label)

    correct = 0
    for labels in clusters.values():
        correct += Counter(labels).most_common(1)[0][1]
    return correct / len(true_labels) if true_labels else 0.0


def silhouette_score_precomputed(matrix: np.ndarray, labels: list[str]) -> float | None:
    """Return silhouette score from a precomputed distance matrix.

    Returns None when fewer than two labels are present, when every sample has
    its own label, or when the matrix shape is invalid.
    """
    distances = np.asarray(matrix, dtype=np.float64)
    sample_count = len(labels)
    if distances.shape != (sample_count, sample_count):
        return None
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2 or len(unique_labels) >= sample_count:
        return None

    scores = []
    for index, label in enumerate(labels):
        same_indices = [candidate for candidate, candidate_label in enumerate(labels) if candidate_label == label and candidate != index]
        other_labels = [candidate_label for candidate_label in unique_labels if candidate_label != label]

        a_value = float(np.mean(distances[index, same_indices])) if same_indices else 0.0
        b_values = []
        for other_label in other_labels:
            other_indices = [candidate for candidate, candidate_label in enumerate(labels) if candidate_label == other_label]
            if other_indices:
                b_values.append(float(np.mean(distances[index, other_indices])))
        if not b_values:
            continue

        b_value = min(b_values)
        denominator = max(a_value, b_value)
        scores.append((b_value - a_value) / denominator if denominator > 0 else 0.0)

    return float(np.mean(scores)) if scores else None


def evaluate_distance_matrix(curves: list[MelodyCurve], matrix: np.ndarray) -> dict[str, float]:
    """Compute label-based metrics directly from a precomputed distance matrix."""
    labels = [curve.label for curve in curves]
    if any(label is None for label in labels):
        return {}

    typed_labels = [str(label) for label in labels]
    if len(set(typed_labels)) < 2:
        return {}

    silhouette = silhouette_score_precomputed(matrix, typed_labels)
    if silhouette is None:
        return {}
    return {
        "silhouette": round(float(silhouette), 4),
    }


def evaluate(curves: list[MelodyCurve], cluster_labels: list[int]) -> dict[str, float]:
    """Compute ARI and purity when every curve has a usable label.

    Returns an empty dict if labels are missing, if fewer than two true labels
    are present, or if the cluster label count does not match the curve count.
    """
    if len(curves) != len(cluster_labels):
        return {}

    true_labels = [curve.label for curve in curves]
    if any(label is None for label in true_labels):
        return {}

    typed_labels = [str(label) for label in true_labels]
    if len(set(typed_labels)) < 2:
        return {}

    return {
        "ari": round(float(_adjusted_rand_score(typed_labels, cluster_labels)), 4),
        "purity": round(float(_compute_purity(typed_labels, cluster_labels)), 4),
    }
