# 音乐旋律线几何相似性分析工具 — 设计文档

**日期**: 2026-06-05
**状态**: 已确认

---

## 1. 项目概述

将音乐旋律线抽象为三维空间曲线（时间、音高、力度），使用 Hausdorff 距离量化不同旋律线之间的几何相似性，并探索该度量方式在区分音乐曲风方面的有效性。

## 2. 产品形态

- **类型**: 桌面交互式应用
- **技术栈**: Python + PySide6 + PyQtGraph (OpenGL 3D) + music21 + scipy + scikit-learn
- **数据源**: MIDI 文件（.mid）
- **曲风**: 数据驱动，不预设类别；可接受用户标签
- **许可**: LGPL（全部依赖均为 LGPL/BSD 兼容）

## 2.1 运行与测试

建议使用 Python 3.11。依赖版本已固定在 `requirements.txt` 中：

```bash
python -m pip install -r requirements.txt
```

启动桌面应用：

```bash
python -m src.main
```

运行测试：

```bash
pytest tests/ -v
```

## 3. 核心功能

1. **批量 MIDI 导入**: 支持多选 .mid 文件，自动解析为旋律曲线；多轨 MIDI 会显示轨道列表，用户可合并所有轨道或选择单一主旋律轨道，并选择最高音线/最强力度线/全部音符提取方式
2. **3D 曲线叠加可视化**: 每条旋律线以不同颜色显示在三维坐标系中（x=时间, y=音高, z=力度），支持旋转/缩放/平移
3. **距离计算**: 提供标准 Hausdorff、Modified Hausdorff、离散 Fréchet 三种算法，构建 N×N 距离矩阵
4. **聚类与评估**: 层次聚类 + MDS/t-SNE 降维可视化；有标签时输出 ARI、纯度指标
5. **导出**: 距离矩阵 CSV、聚类图导出
6. **增量计算**: 已有距离矩阵时新增 MIDI，只计算新增曲线与既有曲线的距离，并动态扩展矩阵；删除曲线或切换归一化会使旧矩阵失效

## 4. 数据模型

### Note (dataclass)

| 字段 | 类型 | 说明 |
|------|------|------|
| timestamp | float | 音符起始时刻（秒） |
| pitch | int | MIDI 音符号 0-127 |
| velocity | int | 力度 0-127 |

### MelodyCurve (dataclass)

| 字段 | 类型 | 说明 |
|------|------|------|
| name | str | 显示名称 |
| filepath | str | 源文件路径 |
| label | str\|None | 曲风标签，可为空 |
| raw_notes | list[Note] | 原始音符 |
| points | np.ndarray | shape (N, 3)，归一化后的 [t, pitch, velocity] |
| color | str | 3D 渲染颜色 |

## 5. 归一化策略

时间轴使用绝对时间（秒），保留节奏信息。提供两种归一化方法，默认使用 Min-Max：

### Min-Max（默认）

| 轴 | 原始范围 | 归一化方法 |
|----|---------|-----------|
| 时间 (x) | 0 ~ T_max | 每条曲线减去起始时间后除以自身总时长 |
| 音高 (y) | 0 ~ 127 | 除以 127 |
| 力度 (z) | 0 ~ 127 | 除以 127 |

所有曲线映射到单位立方体 [0,1]³，便于三维展示。该策略对每条曲线独立稳定，新增 MIDI 不会改变既有曲线坐标，因此支持距离矩阵增量扩展。

### Z-Score（可选）

\[
x' = \frac{x - \mu}{\sigma}
\]

对所有维度做均值 0、标准差 1 的标准化。各维度方差相等，消除量纲差异。当不同维度的分布差异较大时更适用。可在 UI 中切换。

不做 DTW 预对齐，保持几何纯粹性。

## 6. Hausdorff 距离

### 标准 Hausdorff

```
H(A,B) = max( max_a min_b ||a-b||, max_b min_a ||b-a|| )
```

使用 `scipy.spatial.distance.directed_hausdorff` 实现。

### Modified Hausdorff

将 max 替换为 mean，对异常值更鲁棒：

```
h(A,B) = mean_a min_b ||a-b||
H(A,B) = max( h(A,B), h(B,A) )
```

使用 `scipy.spatial.KDTree` 加速最近邻搜索。

### 距离矩阵

对 N 条曲线计算 N×N 对称矩阵，每对计算一次。

## 7. 架构

```
PySide6 Desktop App
├── UI Layer
│   ├── 3D View (PyQtGraph GLViewWidget)
│   ├── File Panel (QListWidget)
│   ├── Distance Matrix (QTableWidget)
│   └── Cluster Panel (matplotlib embedded)
├── Core Layer
│   ├── midi_loader (music21)
│   ├── normalization (numpy)
│   ├── hausdorff (scipy)
│   └── clustering + evaluation (scikit-learn)
└── Data Layer
    ├── MelodyCurve
    └── DistanceMatrix (numpy)
```

## 8. UI 布局

三面板 + 控制栏结构：

- **左侧栏**: MIDI 文件列表，支持勾选显示/隐藏、右键编辑标签/移除
- **中央**: 3D 旋律线叠加视图，鼠标旋转/缩放/平移
- **右侧**: 距离矩阵热力图 + MDS 散点图 + 层次聚类树状图
- **底部控制栏**: 距离算法选择、归一化开关、计算按钮、导出按钮、状态栏

## 9. 项目结构

```
music/
├── src/
│   ├── main.py
│   ├── models/
│   │   ├── note.py
│   │   └── melody_curve.py
│   ├── io/
│   │   └── midi_loader.py
│   ├── processing/
│   │   ├── normalization.py
│   │   └── hausdorff.py
│   ├── analysis/
│   │   ├── distance_matrix.py
│   │   ├── clustering.py
│   │   └── evaluation.py
│   └── ui/
│       ├── main_window.py
│       ├── file_panel.py
│       ├── gl_view.py
│       ├── matrix_panel.py
│       ├── cluster_panel.py
│       └── control_bar.py
├── tests/
│   ├── test_hausdorff.py
│   ├── test_normalization.py
│   ├── test_midi_loader.py
│   ├── test_clustering.py
│   └── fixtures/
├── data/
├── requirements.txt
└── README.md
```

## 10. 依赖

| 包 | 用途 | 许可 |
|----|------|------|
| PySide6 | Qt UI 框架 | LGPL |
| pyqtgraph | 3D OpenGL 可视化 | MIT |
| PyOpenGL | OpenGL 绑定 | BSD |
| music21 | MIDI 解析 | BSD |
| numpy | 数值计算 | BSD |
| scipy | Hausdorff / KDTree | BSD |
| scikit-learn | 聚类 / 降维 / 评估 | BSD |
| matplotlib | 2D 图表（树状图、散点图） | PSF |
| pytest | 测试 | MIT |

## 11. 测试策略

- **单元测试**: 归一化正确性、Hausdorff 与手工计算对比
- **集成测试**: MIDI 加载 → 归一化 → 距离计算完整链路
- **关键断言**: 相同曲线距离=0、平移后距离>0、时间拉伸归一化后距离近似
- **分类验证**: 带标签数据集跑聚类，记录 ARI 指标

## 12. 不做的事项

- 不处理 .wav/.mp3 音频（MVP 仅支持 MIDI；音频转 MIDI 已列入后续扩展）
- 不做 DTW 预对齐
- 不提供在线/Web 部署
- 不预设曲风类别

## 13. 后续可扩展方向

当前 MVP 版本聚焦于旋律线的三维几何特征（时间、音高、力度）。以下方向可在后续版本中引入：

| 扩展应用 | 说明 | 实现思路 |
|----------|------|----------|
| **旋律抄袭/翻唱检测** | 相似旋律在几何空间中是否距离更近，量化借鉴程度 | 已有核心能力直接支持：两两 Hausdorff 距离 → 设定阈值判定相似 |
| **Fréchet 距离对比** | 引入 Fréchet 距离作为对照度量，保留曲线顺序信息 | 离散 Fréchet 算法实现，与 Hausdorff 结果对比分析 |

**附加音乐特征（增强区分力）：**

| 扩展 | 说明 | 实现思路 |
|------|------|----------|
| **和声分析** | 从 MIDI 音轨中统计同时发声的音符组，推断和弦进行（大调/小调、和弦类型分布） | 按时间窗口聚合音符密度 → 和弦模板匹配 → 输出和弦向量作为附加维度的特征 |
| **结构分析** | 检测旋律中的重复、变奏、ABA 等宏观结构 | 自相似矩阵 + 分段检测 → 输出结构相似度分数 |
| **多特征融合** | 将几何 Hausdorff 距离与和声、结构特征结合 | 多核学习或加权融合，联合用于聚类/分类 |
| **音频转 MIDI** | 支持 .wav/.mp3 输入，自动提取旋律转为 MIDI | basic-pitch / CREPE 预训练模型做音高检测，转为 MIDI 后再走标准管线 |

**应用增强功能：**

| 扩展 | 说明 | 实现思路 |
|------|------|----------|
| **多音轨选择** | MIDI 含多轨道时，用户可选择主旋律轨道或合并所有轨道 | music21 解析所有轨道 → UI 列表选择 → 提取指定轨道音符 |
| **维度权重** | 用户可调节时间/音高/力度的权重比例，强调特定维度的相似性 | 归一化后的坐标乘以用户设定的权重系数，再计算距离 |
| **会话保存/加载** | 保存当前加载的曲线、标签、距离矩阵，下次直接恢复 | JSON 存储元数据 + numpy .npz 存储矩阵和曲线数据 |
| **增量计算** | 新增 MIDI 时只计算与已有曲线的距离，不重算整个矩阵 | 距离矩阵动态扩展，新增行/列 |

这些扩展可作为独立的研究方向，在基础工具验证了 Hausdorff 几何方法的有效性后再逐步引入。

## 14. 异常处理

| 场景 | 处理方式 |
|------|----------|
| 非 MIDI 格式文件 | 拒绝加载，弹出提示 |
| MIDI 文件无音符轨道 | 提示该文件无可提取的旋律线 |
| 乐器轨道数为 0 | 检查 `instrument.parts` 为空时跳过 |
| 计算时点集为空 | 跳过该曲线，在状态栏给出警告 |
| 距离矩阵只有 1 条曲线 | 禁用聚类分析，提示至少需要 2 条 |
