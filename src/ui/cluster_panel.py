import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget


try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from scipy.cluster.hierarchy import dendrogram

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    Figure = None
    FigureCanvas = None
    dendrogram = None


class ClusterPanel(QWidget):
    """Right-side tab panel for MDS, dendrogram, and label evaluation."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("聚类结果")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._eval_label = QLabel("（需要曲风标签）")
        self._eval_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        if HAS_MATPLOTLIB:
            self._build_plot_tabs()
        else:
            self._build_fallback_tabs()

    def _build_plot_tabs(self) -> None:
        """Create matplotlib-backed MDS and dendrogram tabs."""
        self._mds_figure = Figure(figsize=(4, 3), dpi=100)
        self._mds_canvas = FigureCanvas(self._mds_figure)
        self._add_tab("MDS", self._mds_canvas)

        self._dendro_figure = Figure(figsize=(4, 3), dpi=100)
        self._dendro_canvas = FigureCanvas(self._dendro_figure)
        self._add_tab("树状图", self._dendro_canvas)

        self._add_tab("评估", self._eval_label)

    def _build_fallback_tabs(self) -> None:
        """Create label-only tabs when matplotlib is unavailable."""
        unavailable = QLabel("matplotlib 不可用，图表视图不可用")
        unavailable.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._add_tab("图表", unavailable)
        self._add_tab("评估", self._eval_label)

    def _add_tab(self, title: str, child: QWidget) -> None:
        """Add a child widget inside a zero-margin tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(child)
        self.tabs.addTab(tab, title)

    def set_result(
        self,
        matrix: np.ndarray,
        names: list[str],
        cluster_result: dict,
        curves: list,
    ) -> None:
        """Render clustering outputs for matrix shape (N, N)."""
        if HAS_MATPLOTLIB:
            self._draw_mds(matrix, names, cluster_result, curves)
            self._draw_dendrogram(cluster_result, names)
        self._show_evaluation(curves, cluster_result)

    def _draw_mds(self, matrix, names, cluster_result, curves) -> None:
        """Draw an MDS scatter plot for a distance matrix."""
        from src.analysis.clustering import mds_reduce

        self._mds_figure.clear()
        axis = self._mds_figure.add_subplot(111)

        if matrix.shape[0] >= 2:
            coords = mds_reduce(matrix, n_components=2)
            colors = _colors_for_curves(curves, cluster_result)
            for index, name in enumerate(names):
                axis.scatter(
                    coords[index, 0],
                    coords[index, 1],
                    c=colors[index],
                    s=40,
                    edgecolors="black",
                    linewidth=0.5,
                )
                axis.annotate(
                    name,
                    (coords[index, 0], coords[index, 1]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    fontsize=7,
                    ha="center",
                )
            axis.set_xlabel("MDS 1")
            axis.set_ylabel("MDS 2")
        else:
            axis.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=axis.transAxes)

        axis.set_title("MDS 降维散点图")
        self._mds_figure.tight_layout()
        self._mds_canvas.draw()

    def _draw_dendrogram(self, cluster_result, names) -> None:
        """Draw a scipy dendrogram when a linkage matrix is available."""
        self._dendro_figure.clear()
        axis = self._dendro_figure.add_subplot(111)

        linkage_matrix = cluster_result.get("linkage")
        if linkage_matrix is not None and len(names) >= 2:
            dendrogram(linkage_matrix, labels=names, ax=axis, leaf_rotation=45, leaf_font_size=8)
            axis.set_title("层次聚类树状图")
        else:
            axis.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=axis.transAxes)

        self._dendro_figure.tight_layout()
        self._dendro_canvas.draw()

    def _show_evaluation(self, curves, cluster_result) -> None:
        """Show ARI and purity when user labels permit evaluation."""
        from src.analysis.evaluation import evaluate

        result = evaluate(curves, cluster_result.get("labels", []))
        if result:
            text = f"ARI: {result['ari']:.4f}  |  纯度: {result['purity']:.4f}"
        else:
            labeled_count = sum(1 for curve in curves if curve.label is not None)
            if labeled_count == 0:
                text = "（无曲风标签）"
            elif labeled_count < len(curves):
                text = "（部分曲线无标签，无法评估）"
            else:
                text = "（标签种类不足，无法评估）"
        self._eval_label.setText(text)


def _colors_for_curves(curves, cluster_result) -> list[str]:
    """Return one display color per curve using labels first, clusters second."""
    palette = [
        "#ff6b6b",
        "#4ecdc4",
        "#ffe66d",
        "#a29bfe",
        "#fd79a8",
        "#00cec9",
        "#fab1a0",
        "#81ecec",
    ]

    if any(curve.label is not None for curve in curves):
        unique_labels = sorted({curve.label for curve in curves if curve.label is not None})
        color_map = {label: palette[index % len(palette)] for index, label in enumerate(unique_labels)}
        return [color_map.get(curve.label, "#999999") for curve in curves]

    labels = cluster_result.get("labels", [0] * len(curves))
    return [palette[int(labels[index]) % len(palette)] for index in range(len(curves))]
