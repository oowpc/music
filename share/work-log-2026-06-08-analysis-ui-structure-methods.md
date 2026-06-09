# 工作日志：可视化分析、结构分析与多方法对比

日期：2026-06-08

## 1. 本次工作目标

本轮工作围绕曲风相似性分析工具继续扩展，重点完成以下功能：

- 将 MDS 可视化、曲风距离矩阵、KNN 分类、混淆矩阵接入 UI；
- 添加旋律结构分析功能，用于检测重复、变奏、ABA 等宏观结构；
- 新增 DTW 距离方法；
- 支持标准 Hausdorff、Modified Hausdorff、离散 Fréchet、DTW 的方法对比；
- 对古典音乐数据进行多距离方法比较，并形成报告；
- 提交并推送代码到 GitHub。

## 2. UI 可视化分析模块

对右侧 `ClusterPanel` 进行了扩展，将原来的聚类结果面板升级为“可视化分析”面板。

新增页签：

- `MDS 可视化`
- `曲风距离矩阵`
- `KNN 分类`
- `混淆矩阵`
- `结构分析`
- `方法对比`

其中：

- MDS 图支持按曲风标签着色，并显示图例；
- 曲风距离矩阵按标签汇总曲风内、曲风间平均距离；
- KNN 分类支持调整 `K` 值，展示 Accuracy、Macro Precision、Macro Recall、Macro F1；
- 混淆矩阵以真实曲风为行、预测曲风为列；
- 结构分析显示片段序列、基础序列、宏观结构、重复次数、变奏次数；
- 方法对比展示四种距离方法的曲风内均值、曲风间均值、区分比、KNN Accuracy 和 Macro-F1。

## 3. 曲风分析与 KNN UI 支撑

新增 `src/analysis/genre_analysis.py`，用于将带曲风标签的曲线和距离矩阵转换为 UI 需要的统计结果。

主要能力：

- 筛选带标签曲线；
- 计算曲风级平均距离矩阵；
- 对带标签样本运行 leave-one-out KNN；
- 输出 KNN 分类指标和混淆矩阵。

该模块不写死曲风类别，因此以后新增更多曲风时不需要修改核心逻辑。

## 4. 旋律结构分析功能

新增 `src/analysis/structure_analysis.py`，实现旋律宏观结构检测。

基本流程：

1. 将旋律曲线按归一化时间切分为固定数量片段；
2. 每个片段内部按自身首尾时间重新归一化；
3. 使用 Modified Hausdorff 比较片段之间的旋律形状；
4. 距离低于重复阈值时判定为“重复”；
5. 距离处于变奏阈值内时判定为“变奏”；
6. 将片段映射为 A、B、C 等结构标签；
7. 根据基础标签序列识别 `ABA`、`重复型`、`通谱型` 等宏观结构。

UI 中新增 `结构分析` 页签，支持调整分段数，默认值为 8。

## 5. DTW 距离方法

在 `src/processing/hausdorff.py` 中新增 `dtw_distance`。

实现特点：

- 使用动态规划计算序列弹性匹配距离；
- 保留旋律点顺序；
- 对累计路径代价按 warping path 长度归一化；
- 更适合处理旋律走势相似但节奏速度不同的情况。

同时将 DTW 接入：

- `src/analysis/distance_matrix.py`
- UI 底部距离算法下拉框
- 片段比较与方法对比分析流程

## 6. 多方法对比

新增 `src/analysis/method_comparison.py`。

对比方法：

- 标准 Hausdorff
- Modified Hausdorff
- 离散 Fréchet
- DTW

输出指标：

- 曲风内平均距离；
- 曲风间平均距离；
- 区分比：曲风间平均距离 / 曲风内平均距离；
- KNN Accuracy；
- Macro-F1；
- 样本数和曲风数。

UI 中新增 `方法对比` 页签，自动比较四种距离方法，并高亮较优的区分比和 KNN Accuracy。

## 7. 古典音乐多方法比较报告

根据 `data/midi_dataset_v1.0/data/raw/classical` 中的古典音乐样本，完成了四种方法的内部相似性比较。

报告文件：

```text
report/classical_method_comparison_report.md
```

实验设置：

- 旋律提取模式：`highest`
- 归一化方式：Min-Max
- 重采样点数：96
- 对比范围：全部 15 首，以及剔除异常样本后的 13 首

主要结论：

- Modified Hausdorff 在古典音乐内部比较中最稳定；
- DTW 能较好处理节奏伸缩，是推荐的对照方法；
- 标准 Hausdorff 对异常片段敏感；
- 离散 Fréchet 保留顺序信息，但整体距离偏严格；
- `braska` 和 `dontbeafraid` 会明显拉大古典音乐内部距离，前期清洗结论有效。

## 8. 测试情况

新增和扩展的测试包括：

- `tests/test_genre_analysis.py`
- `tests/test_cluster_panel_analysis.py`
- `tests/test_structure_analysis.py`
- `tests/test_method_comparison.py`
- `tests/test_hausdorff.py`
- `tests/test_distance_matrix.py`

最终全量测试结果：

```text
87 passed
```

测试中仍有 matplotlib 中文字体缺失警告和 joblib CPU 核心数识别警告，但不影响功能正确性。

## 9. Git 提交记录

本轮已提交并推送到 GitHub。

提交：

```text
ec05a03 feat(analysis): add method and structure comparison UI
```

分支：

```text
codex/task-1-3-scaffold-midi-loader
```

## 10. 后续优化建议

后续可以继续优化以下方向：

- 为 `方法对比` 页签增加 CSV 导出；
- 给 DTW 和 Fréchet 添加下采样参数或 Sakoe-Chiba 窗口约束，提升长曲目计算速度；
- 将结构分析阈值暴露到 UI；
- 在结构分析中支持 AABA、ABAC、Verse-Chorus 等更复杂结构；
- 增加对单曲结构图的可视化，例如用颜色条展示 A/B/A' 段落；
- 将古典音乐多方法比较扩展到全部曲风，形成完整方法评估报告。
