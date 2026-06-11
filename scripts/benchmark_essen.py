#!/usr/bin/env python3
"""Benchmark Essen Folk Song Collection across all four distance methods.

Computes silhouette scores and leave-one-out KNN classification accuracy
for standard, modified, Fréchet and DTW distances on the Essen dataset.
Results are printed to stdout and saved as a Markdown report.
"""

from __future__ import annotations

import logging
import os
import sys
import textwrap

# Ensure the project root is on the import path so `src.*` works when the
# script is invoked directly (e.g. ``python scripts/benchmark_essen.py``).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS, build_matrix
from src.analysis.evaluation import silhouette_score_precomputed
from src.analysis.knn_classifier import leave_one_out_knn, evaluate_predictions
from src.io.essen_loader import download_essen, get_essen_summary, load_essen_collection
from src.processing.normalization import normalize_minmax

logger = logging.getLogger(__name__)

K = 5
MAX_FILES = 200
MIN_SAMPLES_PER_LABEL = 3
METHODS = ["standard", "modified", "frechet", "dtw"]


def _format_pct(value: float) -> str:
    """Return a float formatted as a percentage string."""
    return f"{value * 100:.1f}%"


def _build_results(
    curves: list,
    method: str,
    matrix: np.ndarray,
) -> dict:
    """Compute silhouette and KNN metrics for a distance method."""
    labels = [str(curve.label or "") for curve in curves]
    silhouette = silhouette_score_precomputed(matrix, labels)

    knn_result = None
    knn_accuracy = 0.0
    knn_f1 = 0.0
    effective_k = K

    labeled_indices = [i for i, label in enumerate(labels) if label]
    if len(labeled_indices) >= 2 and len(set(labels[i] for i in labeled_indices)) >= 2:
        labeled_matrix = matrix[np.ix_(labeled_indices, labeled_indices)]
        labeled_labels = [labels[i] for i in labeled_indices]
        effective_k = min(K, len(labeled_indices) - 1)
        if effective_k < 1:
            effective_k = 1
        predictions = leave_one_out_knn(labeled_matrix, labeled_labels, k=effective_k)
        metrics = evaluate_predictions(predictions)
        knn_result = metrics
        knn_accuracy = metrics["accuracy"]
        knn_f1 = metrics["macro_f1"]

    return {
        "method": method,
        "silhouette": silhouette,
        "knn_accuracy": knn_accuracy,
        "knn_f1": knn_f1,
        "k": effective_k,
        "knn_labels": len(set(labels[i] for i in labeled_indices)) if labeled_indices else 0,
    }


def _print_table(results: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    header = f"{'Method':<12} {'Silhouette':>12} {'KNN Acc':>12} {'KNN F1':>12}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for result in results:
        silhouette_str = (
            f"{result['silhouette']:.4f}" if result["silhouette"] is not None else "N/A"
        )
        print(
            f"{result['method']:<12} {silhouette_str:>12} "
            f"{_format_pct(result['knn_accuracy']):>12} "
            f"{_format_pct(result['knn_f1']):>12}"
        )
    print(separator)


def _write_markdown_report(
    results: list[dict],
    summary: dict[str, int],
    curve_count: int,
    filepath: str,
) -> None:
    """Write benchmark results as a Markdown report."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = [
        "# Essen Folk Song Collection Benchmark",
        "",
        f"**Curves loaded**: {curve_count}",
        f"**K for KNN**: {K}",
        "",
        "## Dataset Summary",
        "",
        "| Label | Count |",
        "|-------|-------|",
    ]
    for label, count in sorted(summary.items()):
        lines.append(f"| {label} | {count} |")

    lines.extend([
        "",
        "## Distance Method Comparison",
        "",
        "| Method | Silhouette | KNN Accuracy | KNN Macro F1 |",
        "|--------|------------|-------------|-------------|",
    ])
    for result in results:
        silhouette_str = (
            f"{result['silhouette']:.4f}" if result["silhouette"] is not None else "N/A"
        )
        lines.append(
            f"| {result['method']} | {silhouette_str} "
            f"| {_format_pct(result['knn_accuracy'])} "
            f"| {_format_pct(result['knn_f1'])} |"
        )
    lines.append("")

    with open(filepath, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    logger.info("Report saved to %s", filepath)


def main() -> None:
    """Entry point: load Essen data, compute metrics, print and save results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        essen_dir = download_essen()
    except Exception:
        logger.exception("Failed to obtain Essen dataset")
        sys.exit(1)

    curves = load_essen_collection(essen_dir, max_files=MAX_FILES, shuffle=True)
    if not curves:
        logger.warning("No curves loaded from %s — nothing to benchmark", essen_dir)
        print("No Essen data available. Exiting gracefully.")
        sys.exit(0)

    normalize_minmax(curves)

    from collections import Counter
    label_counts = Counter(c.label for c in curves)
    valid_labels = {l for l, c in label_counts.items() if c >= MIN_SAMPLES_PER_LABEL}
    curves = [c for c in curves if c.label in valid_labels]

    summary = get_essen_summary(curves)

    print(f"\nLoaded {len(curves)} curves across {len(summary)} labels "
          f"(labels with >= {MIN_SAMPLES_PER_LABEL} samples).\n")

    results = []
    for method in METHODS:
        logger.info("Computing %s distance matrix...", method)
        matrix = build_matrix(curves, method=method)
        method_results = _build_results(curves, method, matrix)
        results.append(method_results)

    _print_table(results)
    _write_markdown_report(
        results, summary, len(curves),
        filepath="report/essen_benchmark.md",
    )
    print("\nResults saved to report/essen_benchmark.md")


if __name__ == "__main__":
    main()
