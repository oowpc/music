from collections import Counter, defaultdict

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
