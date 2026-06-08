import numpy as np

from src.analysis.clustering import hierarchical_clustering, mds_reduce, tsne_reduce


def test_hierarchical_clustering_returns_linkage_and_labels():
    matrix = np.array(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.2],
            [0.9, 0.8, 0.2, 0.0],
        ]
    )
    names = ["a", "b", "c", "d"]

    result = hierarchical_clustering(matrix, names)

    assert "linkage" in result
    assert "labels" in result
    assert len(result["labels"]) == 4
    assert result["labels"][0] == result["labels"][1]


def test_mds_reduce_2d():
    matrix = np.array(
        [
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.7],
            [0.8, 0.7, 0.0],
        ]
    )

    coords = mds_reduce(matrix, n_components=2)

    assert coords.shape == (3, 2)


def test_tsne_reduce_2d():
    matrix = np.array(
        [
            [0.0, 0.1, 0.8],
            [0.1, 0.0, 0.7],
            [0.8, 0.7, 0.0],
        ]
    )

    coords = tsne_reduce(matrix, n_components=2)

    assert coords.shape == (3, 2)


def test_hierarchical_clustering_two_items():
    matrix = np.array(
        [
            [0.0, 0.05],
            [0.05, 0.0],
        ]
    )

    result = hierarchical_clustering(matrix, ["x", "y"], n_clusters=2)

    assert result["linkage"] is not None
    assert len(result["labels"]) == 2


def test_hierarchical_clustering_single_item():
    result = hierarchical_clustering(np.array([[0.0]]), ["solo"])

    assert result == {"linkage": None, "labels": [0]}
