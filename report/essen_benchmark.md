# Essen Folk Song Collection Benchmark

**Curves loaded**: 190
**K for KNN**: 5

## Dataset Summary

| Label | Count |
|-------|-------|
| china | 56 |
| deutschl | 126 |
| elsass | 3 |
| oesterrh | 5 |

## Distance Method Comparison

| Method | Silhouette | KNN Accuracy | KNN Macro F1 |
|--------|------------|-------------|-------------|
| standard | -0.0740 | 78.4% | 37.5% |
| modified | -0.0939 | 81.1% | 39.2% |
| frechet | -0.0698 | 79.5% | 38.2% |
| dtw | -0.0908 | 79.5% | 38.4% |
