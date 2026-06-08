from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class KNNPrediction:
    """One KNN prediction with nearest-neighbor details."""

    index: int
    true_label: str
    predicted_label: str
    neighbor_indices: list[int]
    neighbor_labels: list[str]
    neighbor_distances: list[float]


def predict_label(
    distances_to_train: np.ndarray,
    train_labels: list[str],
    k: int = 5,
) -> tuple[str, list[int]]:
    """Predict one label from distances to training samples.

    Returns the predicted label and the local training-set indices of the
    nearest neighbors used for voting.
    """
    distances = np.asarray(distances_to_train, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("distances_to_train must be a 1D array")
    if len(distances) != len(train_labels):
        raise ValueError("distance count must match train label count")
    if not train_labels:
        raise ValueError("train_labels must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")

    neighbor_count = min(k, len(train_labels))
    ordered_indices = sorted(range(len(distances)), key=lambda idx: (distances[idx], str(train_labels[idx])))
    neighbor_indices = ordered_indices[:neighbor_count]
    neighbor_labels = [str(train_labels[idx]) for idx in neighbor_indices]

    vote_counts = Counter(neighbor_labels)
    best_count = max(vote_counts.values())
    candidates = [label for label, count in vote_counts.items() if count == best_count]
    if len(candidates) == 1:
        return candidates[0], neighbor_indices

    mean_distances = {}
    for label in candidates:
        label_distances = [
            distances[idx] for idx in neighbor_indices if str(train_labels[idx]) == label
        ]
        mean_distances[label] = float(np.mean(label_distances))
    predicted = min(candidates, key=lambda label: (mean_distances[label], label))
    return predicted, neighbor_indices


def predict_from_distance_matrix(
    matrix: np.ndarray,
    labels: list[str],
    train_indices: list[int],
    test_indices: list[int],
    k: int = 5,
) -> list[KNNPrediction]:
    """Predict test labels using a precomputed full distance matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("matrix shape must match label count")
    if not train_indices:
        raise ValueError("train_indices must not be empty")

    typed_labels = [str(label) for label in labels]
    predictions = []
    train_labels = [typed_labels[index] for index in train_indices]

    for test_index in test_indices:
        distances = matrix[test_index, train_indices]
        predicted, local_neighbor_indices = predict_label(distances, train_labels, k=k)
        neighbor_indices = [train_indices[index] for index in local_neighbor_indices]
        predictions.append(
            KNNPrediction(
                index=test_index,
                true_label=typed_labels[test_index],
                predicted_label=predicted,
                neighbor_indices=neighbor_indices,
                neighbor_labels=[typed_labels[index] for index in neighbor_indices],
                neighbor_distances=[float(matrix[test_index, index]) for index in neighbor_indices],
            )
        )

    return predictions


def leave_one_out_knn(
    matrix: np.ndarray,
    labels: list[str],
    k: int = 5,
) -> list[KNNPrediction]:
    """Run leave-one-out KNN classification on a precomputed matrix."""
    predictions = []
    for index in range(len(labels)):
        train_indices = [candidate for candidate in range(len(labels)) if candidate != index]
        predictions.extend(
            predict_from_distance_matrix(matrix, labels, train_indices, [index], k=k)
        )
    return predictions


def stratified_split(
    labels: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Return stratified train/test indices for arbitrary label names."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not labels:
        raise ValueError("labels must not be empty")

    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[str(label)].append(index)

    rng = random.Random(random_state)
    train_indices = []
    test_indices = []
    for group_indices in grouped.values():
        shuffled = list(group_indices)
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_size))
        if test_count >= len(shuffled) and len(shuffled) > 1:
            test_count = len(shuffled) - 1
        test_indices.extend(shuffled[:test_count])
        train_indices.extend(shuffled[test_count:])

    train_indices.sort()
    test_indices.sort()
    return train_indices, test_indices


def classification_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    label_order: list[str] | None = None,
) -> dict:
    """Compute accuracy, precision, recall, F1, and confusion matrix."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true and predicted label counts must match")
    if not true_labels:
        raise ValueError("labels must not be empty")

    typed_true = [str(label) for label in true_labels]
    typed_pred = [str(label) for label in predicted_labels]
    labels = label_order or sorted(set(typed_true) | set(typed_pred))
    label_to_index = {label: index for index, label in enumerate(labels)}

    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(typed_true, typed_pred):
        confusion[label_to_index[true_label], label_to_index[pred_label]] += 1

    per_label = {}
    f1_values = []
    precision_values = []
    recall_values = []
    weighted_f1_total = 0.0
    total = len(typed_true)

    for label in labels:
        index = label_to_index[label]
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weighted_f1_total += f1 * support

    accuracy = sum(1 for true, pred in zip(typed_true, typed_pred) if true == pred) / total
    return {
        "labels": labels,
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": weighted_f1_total / total,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def evaluate_predictions(predictions: list[KNNPrediction], label_order: list[str] | None = None) -> dict:
    """Compute classification metrics from KNN prediction records."""
    return classification_metrics(
        [prediction.true_label for prediction in predictions],
        [prediction.predicted_label for prediction in predictions],
        label_order=label_order,
    )
