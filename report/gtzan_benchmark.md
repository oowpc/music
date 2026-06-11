# GTZAN Genre Classification Benchmark

**Curves**: 300 (30 per genre × 10 genres)
**K for KNN**: 5
**Extraction**: highest (monophonic melody from Basic Pitch)
**Normalization**: Min-Max [0,1]^3

## Dataset Summary

| Genre | Count |
|-------|-------|
| blues | 30 |
| classical | 30 |
| country | 30 |
| disco | 30 |
| hiphop | 30 |
| jazz | 30 |
| metal | 30 |
| pop | 30 |
| reggae | 30 |
| rock | 30 |

## Distance Method Comparison

| Method | Silhouette | KNN Acc | Macro Prec | Macro Rec | Macro F1 | Weighted F1 | Time |
|--------|------------|---------|------------|-----------|----------|-------------|------|
| modified | -0.0186 | 43.7% | 51.1% | 43.7% | 42.6% | 42.6% | 7s |
| standard | -0.0382 | 38.7% | 43.2% | 38.7% | 38.5% | 38.5% | 12s |
| frechet | -0.0325 | 25.7% | 27.6% | 25.7% | 23.7% | 23.7% | 1318s |

## Per-Genre Metrics — Modified Hausdorff

| Genre | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| blues | 70.0% | 46.7% | 56.0% | 30 |
| classical | 81.1% | 100.0% | 89.6% | 30 |
| country | 31.2% | 16.7% | 21.7% | 30 |
| disco | 26.7% | 40.0% | 32.0% | 30 |
| hiphop | 88.9% | 26.7% | 41.0% | 30 |
| jazz | 90.9% | 33.3% | 48.8% | 30 |
| metal | 35.2% | 83.3% | 49.5% | 30 |
| pop | 30.6% | 36.7% | 33.3% | 30 |
| reggae | 23.8% | 16.7% | 19.6% | 30 |
| rock | 32.4% | 36.7% | 34.4% | 30 |

## Confusion Matrix (Modified Hausdorff)

| | blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|---|---|---|---|---|---|---|---|---|---|---|
| **blues** | 14 | 0 | 1 | 6 | 0 | 0 | 3 | 2 | 2 | 2 |
| **classical** | 0 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **country** | 1 | 0 | 5 | 6 | 0 | 1 | 3 | 6 | 2 | 6 |
| **disco** | 0 | 0 | 0 | 12 | 0 | 0 | 4 | 5 | 5 | 4 |
| **hiphop** | 1 | 0 | 0 | 2 | 8 | 0 | 14 | 0 | 3 | 2 |
| **jazz** | 2 | 5 | 5 | 0 | 0 | 10 | 0 | 4 | 1 | 3 |
| **metal** | 0 | 0 | 0 | 2 | 1 | 0 | 25 | 0 | 1 | 1 |
| **pop** | 0 | 2 | 2 | 6 | 0 | 0 | 4 | 11 | 2 | 3 |
| **reggae** | 1 | 0 | 1 | 6 | 0 | 0 | 10 | 5 | 5 | 2 |
| **rock** | 1 | 0 | 2 | 5 | 0 | 0 | 8 | 3 | 0 | 11 |

## Note Density per Genre (mean ± std)

| Genre | Notes | Points |
|-------|-------|--------|
| blues | 87 ± 21 | 87 |
| classical | 156 ± 18 | 156 |
| country | 126 ± 24 | 126 |
| disco | 113 ± 29 | 113 |
| hiphop | 42 ± 22 | 42 |
| jazz | 114 ± 21 | 114 |
| metal | 96 ± 38 | 96 |
| pop | 128 ± 20 | 128 |
| reggae | 98 ± 26 | 98 |
| rock | 114 ± 30 | 114 |
