# 工作日志：KNN 曲风分类模块

**日期**: 2026-06-08

## 背景

已有距离矩阵和 MDS 可视化可以展示曲风之间的几何相似性，但还需要一个定量分类实验来验证 Hausdorff 类距离在曲风判别中的有效性。KNN 分类适合直接基于预计算距离矩阵工作，不需要额外特征训练。

## 本次改动

1. 新增 `src/analysis/knn_classifier.py`。
   - `predict_label()`：根据测试样本到训练样本的距离进行 KNN 投票。
   - `predict_from_distance_matrix()`：从完整距离矩阵中选取训练/测试索引并预测。
   - `leave_one_out_knn()`：执行留一法 KNN。
   - `stratified_split()`：按任意标签名称做分层训练/测试切分。
   - `classification_metrics()`：输出 Accuracy、Precision、Recall、F1 和混淆矩阵。

2. 模块不写死曲风类别。
   - 标签来自输入列表。
   - 后续增加新曲风时无需修改 KNN 代码。

3. 补充 `tests/test_knn_classifier.py`。
   - 多数投票。
   - 平票时按平均距离打破平局。
   - 距离矩阵预测。
   - 留一法分类。
   - 分层切分。
   - 分类指标和混淆矩阵。

## 数据集验证

使用过滤后的 56 首样本，距离算法为 Modified Hausdorff，分类方式为留一法 KNN。

不同 k 的结果：

| k | Accuracy | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|
| 1 | 0.5536 | 0.5590 | 0.5536 |
| 3 | 0.5893 | 0.5934 | 0.5886 |
| 5 | 0.6071 | 0.6130 | 0.6077 |
| 7 | 0.6071 | 0.6130 | 0.6077 |

最佳结果为 `k=5` 和 `k=7`，准确率均为 `0.6071`。

`k=5` 混淆矩阵：

| true \ pred | classical | jazz | pop | rock |
|---|---:|---:|---:|---:|
| classical | 10 | 1 | 2 | 0 |
| jazz | 1 | 9 | 2 | 2 |
| pop | 1 | 3 | 7 | 4 |
| rock | 0 | 4 | 2 | 8 |

分类效果：

| 曲风 | Precision | Recall | F1 |
|---|---:|---:|---:|
| classical | 0.8333 | 0.7692 | 0.8000 |
| jazz | 0.5294 | 0.6429 | 0.5806 |
| pop | 0.5385 | 0.4667 | 0.5000 |
| rock | 0.5714 | 0.5714 | 0.5714 |

## 结论

KNN 分类结果明显高于四分类随机猜测的 25%，说明当前旋律线几何距离具有一定曲风判别能力。但准确率约 60.7%，仍不够高。classical 分类表现最好，pop、rock、jazz 之间混淆较明显，这与 MDS 可视化和曲风距离矩阵结论一致。

## 输出文件

```text
results/knn_summary_modified_filtered.csv
results/knn_predictions_modified_filtered.csv
results/knn_confusion_matrix_modified_filtered.csv
```
