#!/usr/bin/env python3
"""Benchmark GTZAN genre classification via MP3/WAV → Hausdorff → KNN.

Transcribes GTZAN audio files through Basic Pitch, normalizes the resulting
melody curves, computes distance matrices for four methods, and evaluates
leave-one-out KNN classification accuracy with per-genre metrics.

Results are printed to stdout and saved as a Markdown report.

Usage (from project root)::

    python scripts/benchmark_gtzan.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from src.analysis.distance_matrix import DISTANCE_FUNCTIONS, build_matrix
from src.analysis.evaluation import silhouette_score_precomputed
from src.analysis.knn_classifier import (
    classification_metrics,
    leave_one_out_knn,
)
from src.io.audio_loader import load_audio
from src.processing.normalization import normalize_minmax

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GTZAN_DIR = Path(_project_root) / "data" / "audio" / "gtzan"
CACHE_PATH = GTZAN_DIR / ".transcription_cache_v1.npz"  # serialised curves
REPORT_PATH = Path(_project_root) / "report" / "gtzan_benchmark.md"

K = 5
MAX_PER_GENRE = 30  # 0 = all
METHODS = ["standard", "modified", "frechet", "dtw"]

GENRE_ORDER = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _load_or_transcribe(
    gtzan_dir: Path,
    max_per_genre: int,
    cache_path: Path,
) -> tuple[list, dict[str, int]]:
    """Load GTZAN WAVs, transcribing via Basic Pitch with on-disk cache.

    Returns ``(curves, skipped_by_genre)``.
    """
    if cache_path.exists():
        logger.info("Loading cached transcriptions from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        curves_data = data["curves"].tolist()
        from src.models.note import Note
        from src.models.melody_curve import MelodyCurve

        curves = []
        for entry in curves_data:
            raw_notes = [Note(t, p, v) for t, p, v in entry["raw_notes"]]
            c = MelodyCurve(
                name=entry["name"],
                filepath=entry["filepath"],
                label=entry["label"],
                raw_notes=raw_notes,
                color=entry.get("color", "#ffffff"),
            )
            curves.append(c)
        skipped = data.get("skipped_by_genre", {}).tolist() if "skipped_by_genre" in data else {}
        return curves, skipped

    from src.models.note import Note
    from src.models.melody_curve import MelodyCurve

    logger.info("Transcribing GTZAN audio (max %d per genre)…", max_per_genre)
    curves = []
    skipped_by_genre: dict[str, int] = {}

    for genre in GENRE_ORDER:
        genre_dir = gtzan_dir / genre
        if not genre_dir.is_dir():
            continue
        wavs = sorted(genre_dir.glob("*.wav"))
        if max_per_genre > 0:
            wavs = wavs[:max_per_genre]
        skipped = 0
        for wav in wavs:
            curve = load_audio(str(wav), extraction_mode="highest")
            if curve is None:
                skipped += 1
                continue
            curve.label = genre
            curve.color = "#ffffff"
            # Store raw as (timestamp, pitch, velocity) tuples for caching
            curve._cacheable = [(n.timestamp, n.pitch, n.velocity) for n in curve.raw_notes]
            curves.append(curve)
        if skipped:
            skipped_by_genre[genre] = skipped
        load_audio.__module__  # suppress unused warning

    # Write cache
    cache_entries = []
    for c in curves:
        cache_entries.append({
            "name": c.name,
            "filepath": c.filepath,
            "label": c.label,
            "color": c.color,
            "raw_notes": getattr(c, "_cacheable", [(n.timestamp, n.pitch, n.velocity) for n in c.raw_notes]),
        })
    np.savez(cache_path, curves=np.array(cache_entries, dtype=object),
             skipped_by_genre=np.array(skipped_by_genre, dtype=object))
    # Clean temp attrs
    for c in curves:
        if hasattr(c, "_cacheable"):
            del c._cacheable
    logger.info("Cached %d curves to %s", len(curves), cache_path)
    return curves, skipped_by_genre


def _build_results(
    curves: list,
    method: str,
    matrix: np.ndarray,
) -> dict:
    """Evaluate one distance method against ground-truth genre labels."""
    labels = [str(c.label or "") for c in curves]
    silhouette = silhouette_score_precomputed(matrix, labels)

    predictions = leave_one_out_knn(matrix, labels, k=K)
    metrics = classification_metrics(
        [p.true_label for p in predictions],
        [p.predicted_label for p in predictions],
        label_order=GENRE_ORDER,
    )

    return {
        "method": method,
        "silhouette": silhouette,
        "knn_accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_label": metrics["per_label"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "label_order": metrics["labels"],
        "k": K,
    }


def _print_table(results: list[dict]) -> None:
    header = f"{'Method':<10} {'Silhouette':>10} {'KNN Acc':>10} {'Macro F1':>10} {'Time':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        sil = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "N/A"
        print(
            f"{r['method']:<10} {sil:>10} "
            f"{_format_pct(r['knn_accuracy']):>10} "
            f"{_format_pct(r['macro_f1']):>10} "
            f"{r.get('elapsed', 0):>7.0f}s"
        )
    print(sep)


def _write_markdown_report(
    results: list[dict],
    summary: dict[str, int],
    curve_count: int,
    skipped: dict[str, int],
    max_per_genre: int,
    filepath: str,
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    lines = [
        "# GTZAN Genre Classification Benchmark",
        "",
        f"**Curves loaded**: {curve_count}",
        f"**Max per genre**: {max_per_genre if max_per_genre > 0 else 'all'}",
        f"**K for KNN**: {K}",
        f"**Extraction mode**: highest (monophonic melody)",
        f"**Transcription**: Basic Pitch (Spotify ICASSP 2022)",
        "",
        "## Dataset Summary",
        "",
        "| Genre | Count |",
        "|-------|-------|",
    ]
    for genre in GENRE_ORDER:
        count = summary.get(genre, 0)
        note = f" ({skipped[genre]} failed)" if genre in skipped else ""
        lines.append(f"| {genre} | {count}{note} |")

    lines.extend([
        "",
        "## Distance Method Comparison",
        "",
        "| Method | Silhouette | KNN Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Time |",
        "|--------|------------|-------------|-----------------|--------------|----------|-------------|------|",
    ])
    for r in results:
        sil = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "N/A"
        lines.append(
            f"| {r['method']} | {sil} "
            f"| {_format_pct(r['knn_accuracy'])} "
            f"| {_format_pct(r['macro_precision'])} "
            f"| {_format_pct(r['macro_recall'])} "
            f"| {_format_pct(r['macro_f1'])} "
            f"| {_format_pct(r['weighted_f1'])} "
            f"| {r.get('elapsed', 0):.0f}s |"
        )

    # Best method's per-genre breakdown
    if results:
        best = max(results, key=lambda r: r.get("macro_f1", 0) or 0)
        lines.extend([
            "",
            f"## Per-Genre Metrics — {best['method'].title()} Hausdorff",
            "",
            "| Genre | Precision | Recall | F1 | Support |",
            "|-------|-----------|--------|----|---------|",
        ])
        for genre in best.get("label_order", []):
            pl = best.get("per_label", {}).get(genre, {})
            lines.append(
                f"| {genre} "
                f"| {_format_pct(pl.get('precision', 0))} "
                f"| {_format_pct(pl.get('recall', 0))} "
                f"| {_format_pct(pl.get('f1', 0))} "
                f"| {pl.get('support', 0)} |"
            )

        # Confusion matrix
        lines.extend([
            "",
            "## Confusion Matrix",
            "",
        ])
        cm = best.get("confusion_matrix", [])
        label_order = best.get("label_order", [])
        if cm and label_order:
            # Header
            header_cells = "| | " + " | ".join(label_order) + " |"
            lines.append(header_cells)
            sep = "|---|" + "|".join(["---" for _ in label_order]) + "|"
            lines.append(sep)
            for i, label in enumerate(label_order):
                row = [str(cm[i][j]) for j in range(len(label_order))]
                lines.append(f"| **{label}** | " + " | ".join(row) + " |")

    lines.append("")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report saved to %s", filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --- Load / transcribe ---
    curves, skipped = _load_or_transcribe(GTZAN_DIR, MAX_PER_GENRE, CACHE_PATH)
    if not curves:
        logger.warning("No curves loaded — nothing to benchmark")
        sys.exit(0)

    normalize_minmax(curves)
    summary = dict(Counter(c.label for c in curves))

    print(f"\nLoaded {len(curves)} curves across {len(summary)} genres.\n")

    results = []
    for method in METHODS:
        logger.info("Computing %s distance matrix…", method)
        t0 = time.time()
        matrix = build_matrix(curves, method=method)
        elapsed = time.time() - t0
        method_results = _build_results(curves, method, matrix)
        method_results["elapsed"] = elapsed
        results.append(method_results)
        logger.info("  %s: acc=%s  F1=%s  (%.0fs)", method,
                     _format_pct(method_results["knn_accuracy"]),
                     _format_pct(method_results["macro_f1"]),
                     elapsed)

    _print_table(results)
    _write_markdown_report(
        results, summary, len(curves), skipped, MAX_PER_GENRE,
        filepath=str(REPORT_PATH),
    )
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
