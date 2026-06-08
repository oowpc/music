# 古典音乐多距离方法对比分析报告

## 1. 分析目的

本次实验针对数据集中的古典音乐样本，比较以下四种旋律曲线距离方法对古典音乐内部相似性的刻画效果：

- 标准 Hausdorff 距离
- Modified Hausdorff 距离
- 离散 Fréchet 距离
- DTW 距离

目标是判断不同方法在古典音乐样本上的稳定性、敏感性，以及它们对后续曲风区分任务的适用性。

## 2. 数据与预处理

数据路径：

```text
data/midi_dataset_v1.0/data/raw/classical
```

共读取古典 MIDI 文件 15 首：

| 文件 | 音符数 | 时长/秒 |
|---|---:|---:|
| BlueStone_LastDungeon | 478 | 150.00 |
| braska | 187 | 132.00 |
| caitsith | 774 | 329.29 |
| Cids | 246 | 131.25 |
| cosmo | 473 | 187.85 |
| costadsol | 195 | 63.81 |
| dayafter | 426 | 121.91 |
| decisive | 550 | 104.01 |
| dontbeafraid | 462 | 168.96 |
| DOS | 476 | 146.39 |
| thenightmarebegins | 672 | 118.50 |
| thoughts | 354 | 114.86 |
| tifap | 282 | 129.17 |
| tpirtsd-piano | 270 | 108.00 |
| traitor | 212 | 93.91 |

预处理方式：

- 使用 `music21` 解析 MIDI；
- 使用 `highest` 模式提取主旋律线，即同一时间点保留最高音；
- 使用 Min-Max 归一化；
- 为避免 Fréchet 与 DTW 在长序列上计算过慢，将每首曲线统一重采样到 96 个旋律点；
- 分别统计全部 15 首，以及剔除明显异常样本后的 13 首结果。

剔除样本：

| 文件 | 剔除原因 |
|---|---|
| braska | 前期数据清洗中被标记为古典内部距离异常点 |
| dontbeafraid | 前期数据清洗中被标记为古典内部距离异常点 |

## 3. 方法说明

### 3.1 标准 Hausdorff 距离

标准 Hausdorff 距离关注两个点集之间的最大最近邻距离。它对局部离群点非常敏感，如果一首曲子中存在少量偏离主体旋律的片段，距离会被显著放大。

### 3.2 Modified Hausdorff 距离

Modified Hausdorff 使用平均最近邻距离代替最大最近邻距离，因此更能反映整体旋律形状的平均差异，对局部离群点更稳健。

### 3.3 离散 Fréchet 距离

离散 Fréchet 距离保留曲线点的先后顺序，适合比较旋律发展路径是否一致。但因为它强调顺序匹配，距离通常比 Modified Hausdorff 更大。

### 3.4 DTW 距离

DTW 允许两个序列在时间轴上进行弹性对齐，适合处理节奏速度不同但旋律走向相似的情况。本项目中使用归一化路径代价，使不同长度曲线的结果更可比。

## 4. 全部 15 首古典音乐内部比较

| 方法 | 平均距离 | 中位数 | 标准差 |
|---|---:|---:|---:|
| 标准 Hausdorff | 0.307370 | 0.213441 | 0.226901 |
| Modified Hausdorff | 0.126505 | 0.097962 | 0.099565 |
| 离散 Fréchet | 0.342667 | 0.260546 | 0.219434 |
| DTW | 0.147376 | 0.125042 | 0.100136 |

### 最近曲目对

| 方法 | 最近曲目对 | 距离 |
|---|---|---:|
| 标准 Hausdorff | Cids - thenightmarebegins | 0.073930 |
| Modified Hausdorff | Cids - dayafter | 0.025785 |
| 离散 Fréchet | cosmo - dayafter | 0.110236 |
| DTW | Cids - dayafter | 0.034872 |

### 最远曲目对

| 方法 | 最远曲目对 | 距离 |
|---|---|---:|
| 标准 Hausdorff | cosmo - dontbeafraid | 0.841072 |
| Modified Hausdorff | braska - caitsith | 0.321706 |
| 离散 Fréchet | BlueStone_LastDungeon - dontbeafraid | 0.855772 |
| DTW | BlueStone_LastDungeon - braska | 0.340422 |

## 5. 清洗后 13 首古典音乐内部比较

| 方法 | 平均距离 | 中位数 | 标准差 |
|---|---:|---:|---:|
| 标准 Hausdorff | 0.185309 | 0.187843 | 0.055960 |
| Modified Hausdorff | 0.075082 | 0.053727 | 0.041755 |
| 离散 Fréchet | 0.225461 | 0.224470 | 0.061642 |
| DTW | 0.095309 | 0.074799 | 0.043732 |

### 最近曲目对

| 方法 | 最近曲目对 | 距离 |
|---|---|---:|
| 标准 Hausdorff | Cids - thenightmarebegins | 0.073930 |
| Modified Hausdorff | Cids - dayafter | 0.025785 |
| 离散 Fréchet | cosmo - dayafter | 0.110236 |
| DTW | Cids - dayafter | 0.034872 |

### 最远曲目对

| 方法 | 最远曲目对 | 距离 |
|---|---|---:|
| 标准 Hausdorff | BlueStone_LastDungeon - tpirtsd-piano | 0.309983 |
| Modified Hausdorff | thenightmarebegins - tifap | 0.152797 |
| 离散 Fréchet | BlueStone_LastDungeon - tifap | 0.370665 |
| DTW | caitsith - tifap | 0.169023 |

## 6. 结果分析

### 6.1 异常样本影响

剔除 `braska` 和 `dontbeafraid` 后，四种方法的平均距离都明显下降：

| 方法 | 全部15首均值 | 清洗后13首均值 | 下降幅度 |
|---|---:|---:|---:|
| 标准 Hausdorff | 0.307370 | 0.185309 | 39.71% |
| Modified Hausdorff | 0.126505 | 0.075082 | 40.65% |
| 离散 Fréchet | 0.342667 | 0.225461 | 34.21% |
| DTW | 0.147376 | 0.095309 | 35.33% |

这说明前期识别出的异常样本确实会显著拉大古典音乐内部距离，影响曲风内部一致性的估计。

### 6.2 方法稳定性

清洗后结果中：

- Modified Hausdorff 的平均距离最低，为 0.075082；
- DTW 次之，为 0.095309；
- 标准 Hausdorff 与离散 Fréchet 距离更高；
- Modified Hausdorff 和 DTW 的标准差也较低，说明结果更稳定。

这表明 Modified Hausdorff 和 DTW 更适合用于描述古典音乐内部的整体相似性。

### 6.3 方法差异解释

标准 Hausdorff 距离容易受到局部离群点影响，因此同为古典音乐时，如果某首曲子有较高或较低的局部旋律片段，距离会被放大。

Modified Hausdorff 使用平均最近邻距离，能更稳定地反映整体旋律几何形状，因此在古典音乐内部比较中表现最好。

离散 Fréchet 保留旋律顺序，对旋律发展路径的变化更敏感，因此距离整体偏大。它适合用于判断两段旋律是否具有相近的发展顺序，但不一定适合作为宽松的曲风相似度指标。

DTW 可以对齐节奏速度不同的旋律片段，因此当两首曲子旋律走势接近但节奏伸缩不同，DTW 能给出较低距离。它比 Fréchet 更宽容，也比标准 Hausdorff 更稳定。

## 7. 结论

对古典音乐内部相似性分析而言：

| 推荐程度 | 方法 | 原因 |
|---|---|---|
| 首选 | Modified Hausdorff | 稳定、平均距离最低、对异常点不敏感 |
| 推荐 | DTW | 能处理节奏伸缩，适合旋律走势相似但速度不同的曲子 |
| 辅助 | 离散 Fréchet | 适合分析旋律顺序是否一致，但距离偏严格 |
| 辅助 | 标准 Hausdorff | 可用于发现离群片段，但不适合作为主要曲风距离 |

因此，后续曲风区分实验中，建议以 Modified Hausdorff 作为主指标，以 DTW 作为对照指标；Fréchet 可用于分析旋律顺序结构，标准 Hausdorff 更适合用于异常检测。

## 8. 后续工作

后续可以继续做三类扩展：

- 在全部曲风上比较四种方法的曲风区分能力；
- 在 UI 的“方法对比”页签中加入导出 CSV 功能；
- 对 DTW 和 Fréchet 加入下采样或 Sakoe-Chiba 窗口约束，提高长曲目计算速度。
