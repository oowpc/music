import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def _condensed_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the upper triangle of a square distance matrix as shape (K,)."""
    return np.asarray(matrix, dtype=np.float64)[np.triu_indices(matrix.shape[0], k=1)]


def hierarchical_clustering(
    matrix: np.ndarray,
    names: list[str],
    n_clusters: int = 2,
) -> dict:
    """Cluster a precomputed distance matrix with hierarchical clustering.

    Returns a dict containing ``linkage`` as a scipy linkage matrix or None,
    and ``labels`` as one integer cluster label per input name.
    """
    item_count = len(names)
    if item_count < 2:
        return {"linkage": None, "labels": [0] * item_count}

    if matrix.shape != (item_count, item_count):
        raise ValueError("matrix shape must match the number of names")

    condensed = _condensed_distance_matrix(matrix)
    linkage_matrix = linkage(condensed, method="ward")
    labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    return {
        "linkage": linkage_matrix,
        "labels": [int(label) for label in labels],
    }


def mds_reduce(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce a precomputed distance matrix to coordinates with shape (N, D)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    sample_count = matrix.shape[0]
    if sample_count < 2:
        return np.zeros((sample_count, n_components), dtype=np.float64)

    squared = matrix**2
    centering = np.eye(sample_count) - np.ones((sample_count, sample_count)) / sample_count
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]

    coords = np.zeros((sample_count, n_components), dtype=np.float64)
    for output_axis, eigen_index in enumerate(order[:n_components]):
        eigenvalue = max(float(eigenvalues[eigen_index]), 0.0)
        coords[:, output_axis] = eigenvectors[:, eigen_index] * np.sqrt(eigenvalue)
    return coords


def tsne_reduce(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce a precomputed distance matrix with t-SNE to shape (N, D)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    sample_count = matrix.shape[0]
    if sample_count < 3:
        return mds_reduce(matrix, n_components)

    try:
        from sklearn.manifold import TSNE

        model = TSNE(
            n_components=n_components,
            metric="precomputed",
            init="random",
            random_state=42,
            perplexity=max(1, min(30, sample_count - 1)),
        )
        return model.fit_transform(matrix)
    except Exception:
        return mds_reduce(matrix, n_components)
