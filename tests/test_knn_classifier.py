import numpy as np
import pytest

from src.analysis.knn_classifier import (
    classification_metrics,
    leave_one_out_knn,
    predict_from_distance_matrix,
    predict_label,
    stratified_split,
)


def test_predict_label_uses_majority_vote():
    distances = np.array([0.1, 0.2, 0.3, 0.4])
    labels = ["pop", "rock", "pop", "jazz"]

    predicted, neighbors = predict_label(distances, labels, k=3)

    assert predicted == "pop"
    assert neighbors == [0, 1, 2]


def test_predict_label_breaks_ties_by_mean_distance():
    distances = np.array([0.1, 0.2, 0.11, 0.9])
    labels = ["pop", "rock", "rock", "pop"]

    predicted, neighbors = predict_label(distances, labels, k=4)

    assert predicted == "rock"
    assert neighbors == [0, 2, 1, 3]


def test_predict_from_distance_matrix_returns_neighbor_details():
    matrix = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.8],
            [0.9, 0.8, 0.0],
        ]
    )
    labels = ["pop", "pop", "jazz"]

    predictions = predict_from_distance_matrix(matrix, labels, [0, 2], [1], k=1)

    assert predictions[0].true_label == "pop"
    assert predictions[0].predicted_label == "pop"
    assert predictions[0].neighbor_indices == [0]


def test_leave_one_out_knn_classifies_clustered_matrix():
    matrix = np.array(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.2],
            [0.9, 0.8, 0.2, 0.0],
        ]
    )
    labels = ["pop", "pop", "rock", "rock"]

    predictions = leave_one_out_knn(matrix, labels, k=1)

    assert [prediction.predicted_label for prediction in predictions] == labels


def test_stratified_split_keeps_each_label_in_test_set():
    labels = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]

    train_indices, test_indices = stratified_split(labels, test_size=0.34, random_state=1)

    assert train_indices
    assert test_indices
    assert {labels[index] for index in test_indices} == {"a", "b", "c"}


def test_classification_metrics_returns_confusion_and_scores():
    result = classification_metrics(
        ["pop", "pop", "rock", "rock"],
        ["pop", "rock", "rock", "rock"],
        label_order=["pop", "rock"],
    )

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["confusion_matrix"].tolist() == [[1, 1], [0, 2]]
    assert result["per_label"]["pop"]["recall"] == pytest.approx(0.5)
    assert result["per_label"]["rock"]["precision"] == pytest.approx(2 / 3)


def test_predict_label_validates_input():
    with pytest.raises(ValueError):
        predict_label(np.array([]), [], k=1)
    with pytest.raises(ValueError):
        predict_label(np.array([0.1]), ["pop"], k=0)
