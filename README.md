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

## 3. 核心功能

1. **批量 MIDI 导入**: 支持多选 .mid 文件，自动解析为旋律曲线
2. **3D 曲线叠加可视化**: 每条旋律线以不同颜色显示在三维坐标系中（x=时间, y=音高, z=力度），支持旋转/缩放/平移
3. **Hausdorff 距离计算**: 提供标准 Hausdorff 和 Modified Hausdorff 两种算法，构建 N×N 距离矩阵
4. **聚类与评估**: 层次聚类 + MDS/t-SNE 降维可视化；有标签时输出 ARI、纯度指标
5. **导出**: 距离矩阵 CSV、聚类图导出

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

时间轴使用绝对时间（秒），保留节奏信息。三条轴统一归一化到 [0, 1]：

| 轴 | 原始范围 | 归一化方法 |
|----|---------|-----------|
| 时间 (x) | 0 ~ T_max | 除以总时长 |
| 音高 (y) | 0 ~ 127 | 除以 127 |
| 力度 (z) | 0 ~ 127 | 除以 127 |

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

- 不处理 .wav/.mp3 音频（不做音高检测）
- 不做 DTW 预对齐
- 不提供在线/Web 部署
- 不预设曲风类别
