# 工作日志：v1.1 标准集、单曲识别、旋律检测与评价补全

日期：2026-06-09

## 1. 本次工作目标

本轮工作围绕新增数据集 `midi_dataset_v1.1` 继续完善项目，主要目标包括：

- 使用 v1.1 数据集评估当前方法效果；
- 筛选并隔离明显不合格 MIDI；
- 将清洗后的曲目作为标准集；
- 在 UI 中实现输入一首 MIDI 输出曲风；
- 增加单曲识别结果面板；
- 实现旋律抄袭/翻唱检测的最小版本；
- 补全层次聚类树状图和轮廓系数；
- 提交并推送相关改动。

## 2. v1.1 数据集评估

数据路径：

```text
data/midi_dataset_v1.1/data/raw
```

原始数据包含 5 个曲风：

| 曲风 | 数量 |
|---|---:|
| classical | 25 |
| folk | 25 |
| jazz | 25 |
| pop | 25 |
| rock | 25 |

初始评估中，`rock/AchillesLastStand.mid` 解析耗时异常，因此先跳过，实际评估 124 首。

评估设置：

- 主旋律提取：`highest`
- 归一化：Min-Max
- 重采样：64 点
- 主距离：Modified Hausdorff
- 分类方法：leave-one-out KNN

主要结果：

| 指标 | 数值 |
|---|---:|
| 曲风内平均距离 | 0.0910 |
| 曲风间平均距离 | 0.2144 |
| 区分比 | 2.355 |
| KNN 最佳准确率 | 79.84% |

结论：v1.1 的曲风区分度明显优于 v1.0。

## 3. 数据清洗与隔离

根据以下规则筛选明显不合格样本：

- 解析异常或解析超慢；
- 音符数少于 100；
- 时长短于 30 秒；
- 同曲风内部距离离群。

共移动 9 首 MIDI 到隔离目录：

```text
data/midi_dataset_v1.1/data/excluded_quality_2026-06-09
```

隔离记录：

```text
results/v1_1/excluded_quality_2026-06-09.csv
```

被隔离文件：

| 曲风 | 文件 | 原因 |
|---|---|---|
| folk | ashover22.mid | 音符过少、时长过短 |
| folk | ashover31.mid | 同曲风距离离群 |
| jazz | AliceInWonderland.mid | 同曲风距离离群 |
| jazz | Brazillike.mid | 同曲风距离离群 |
| jazz | Desafinado.mid | 同曲风距离离群 |
| jazz | Effendi - McCoy Tyner.mid | 同曲风距离离群 |
| rock | AchillesLastStand.mid | 解析超慢 |
| rock | AllMyLove.mid | 同曲风距离离群 |
| rock | BlackDog.mid | 同曲风距离离群 |

清洗后标准集数量：

| 曲风 | 数量 |
|---|---:|
| classical | 25 |
| folk | 23 |
| jazz | 21 |
| pop | 25 |
| rock | 22 |
| 合计 | 116 |

清洗后评估：

| 指标 | 清洗前 | 清洗后 |
|---|---:|---:|
| 曲风内平均距离 | 0.0910 | 0.0861 |
| 曲风间平均距离 | 0.2144 | 0.2119 |
| 区分比 | 2.355 | 2.461 |
| KNN 最佳准确率 | 79.84% | 79.31% |

结论：清洗后类内更紧，区分比提升，KNN 准确率基本稳定。

## 4. 标准集缓存与单曲曲风识别

新增模块：

```text
src/analysis/reference_classifier.py
```

功能：

- 从清洗后的 v1.1 raw 数据构建标准集；
- 对标准集曲线统一归一化并重采样到 64 点；
- 保存标准集缓存；
- 输入一首 MIDI 后，使用 KNN 返回预测曲风和最近邻。

标准集缓存文件：

```text
results/v1_1/standard_set_resampled64.npz
```

UI 新增按钮：

```text
识别曲风
```

识别流程：

1. 用户选择一首 MIDI；
2. 系统使用 `highest` 模式提取旋律；
3. Min-Max 归一化；
4. 重采样到 64 点；
5. 使用当前 UI 选择的距离方法和 K 值；
6. 输出预测曲风和最近邻。

## 5. 单曲识别结果面板

右侧 `可视化分析` 面板新增页签：

```text
单曲识别
```

展示内容：

- 输入曲目名；
- 预测曲风；
- 置信度；
- Top 最近邻；
- 最近邻曲名、曲风、距离；
- 曲风投票数；
- 每个候选曲风的平均距离。

这样识别结果不再只显示在状态栏中，便于解释和展示。

## 6. 旋律抄袭/翻唱检测最小版本

新增模块：

```text
src/analysis/melody_similarity.py
```

UI 新增按钮：

```text
旋律检测
```

右侧新增页签：

```text
旋律检测
```

最小版本功能：

- 选择两首 MIDI；
- 分别提取主旋律；
- 统一归一化并重采样；
- 计算全曲 Modified Hausdorff 距离；
- 计算全曲 DTW 距离；
- 搜索最相似片段；
- 输出相似等级和相似度分数。

判定等级：

- 高度相似；
- 疑似翻唱/借鉴；
- 局部片段相似；
- 不相似。

输出内容：

- 两首曲名；
- 判定等级；
- 相似度分数；
- Modified Hausdorff 距离；
- DTW 距离；
- 最相似片段时间范围；
- 最相似片段距离。

## 7. 距离方法切换自动刷新

修复了 UI 中切换距离算法后图像不变的问题。

原因：

- `ControlBar` 已经发出 `method_changed` 信号；
- 但 `MainWindow` 原先没有连接该信号；
- 因此下拉框改变后不会自动重算距离矩阵和 MDS。

修复后：

- 如果已有距离矩阵，切换距离方法后自动重新计算；
- 距离矩阵、MDS、分析页签会同步刷新；
- 如果还没计算过矩阵，则提示用户重新计算。

## 8. 树状图与轮廓系数

补全了项目提纲中的两个缺口。

### 8.1 层次聚类树状图

右侧 `可视化分析` 面板新增：

```text
树状图
```

使用当前距离矩阵的层次聚类 linkage 结果绘制 dendrogram。

### 8.2 轮廓系数

新增基于预计算距离矩阵的轮廓系数计算：

```text
silhouette_score_precomputed
```

右侧新增：

```text
定量评价
```

展示指标：

- ARI；
- 纯度；
- 轮廓系数；
- 类内平均距离；
- 类间平均距离；
- 类间/类内比值。

## 9. 生成报告与结果文件

新增报告：

```text
report/midi_dataset_v1_1_evaluation_report.md
```

新增结果目录：

```text
results/v1_1
```

主要结果文件：

- `distance_matrix_modified_cleaned_resampled64.csv`
- `genre_distance_matrix_modified_cleaned_resampled64.csv`
- `knn_summary_modified_cleaned_resampled64.csv`
- `knn_confusion_matrix_modified_cleaned_resampled64_k5.csv`
- `method_comparison_sample10_resampled64.csv`
- `mds_modified_resampled64_skip_slow.png`
- `standard_set_resampled64.npz`
- `excluded_quality_2026-06-09.csv`

## 10. 测试情况

新增或更新测试：

- `tests/test_reference_classifier.py`
- `tests/test_main_window_reference_classification.py`
- `tests/test_melody_similarity.py`
- `tests/test_main_window_similarity_detection.py`
- `tests/test_main_window_method_change.py`
- `tests/test_evaluation.py`
- `tests/test_cluster_panel_analysis.py`

最终测试结果：

```text
101 passed
```

测试中仍存在 matplotlib 中文字体缺失警告和 joblib CPU 核心数识别警告，但不影响功能正确性。

## 11. Git 提交记录

本轮相关提交：

```text
12ffece feat(ui): add reference classification and melody similarity detection
35eedf6 feat(analysis): add dendrogram and silhouette metrics
```

分支：

```text
codex/task-1-3-scaffold-midi-loader
```

## 12. 后续建议

后续可以继续优化：

- 给“旋律检测”增加阈值调节 UI；
- 增加 Top-N 相似歌曲检索；
- 给标准集缓存增加“重新生成”按钮；
- 对 DTW 和 Fréchet 增加窗口约束，提高长曲目计算速度；
- 增加结构图可视化，例如 A/B/A' 彩色时间条；
- 将批量评估流程做成 UI 功能；
- 使用更丰富的音乐特征减少 `classical/jazz` 与 `pop/rock` 混淆。
