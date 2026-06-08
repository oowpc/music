import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.models.melody_curve import MelodyCurve
from src.ui.cluster_panel import ClusterPanel


def _app():
    """Return a QApplication singleton for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _curve(name: str, label: str) -> MelodyCurve:
    """Create a labeled curve for cluster-panel tests."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        label=label,
        points=np.array(
            [
                (0.00, 0.40, 0.7),
                (0.15, 0.50, 0.7),
                (0.30, 0.60, 0.7),
                (0.36, 0.80, 0.7),
                (0.50, 0.70, 0.7),
                (0.63, 0.85, 0.7),
                (0.70, 0.40, 0.7),
                (0.85, 0.50, 0.7),
                (1.00, 0.60, 0.7),
            ],
            dtype=float,
        ),
    )


def test_cluster_panel_shows_requested_analysis_tabs():
    _app()
    panel = ClusterPanel()

    titles = [panel.tabs.tabText(index) for index in range(panel.tabs.count())]

    assert "MDS 可视化" in titles
    assert "曲风距离矩阵" in titles
    assert "KNN 分类" in titles
    assert "混淆矩阵" in titles
    assert "结构分析" in titles
    assert "方法对比" in titles


def test_cluster_panel_renders_genre_knn_and_confusion_tables():
    _app()
    panel = ClusterPanel()
    matrix = np.array(
        [
            [0.0, 0.1, 1.0, 1.1],
            [0.1, 0.0, 0.9, 1.0],
            [1.0, 0.9, 0.0, 0.2],
            [1.1, 1.0, 0.2, 0.0],
        ]
    )
    curves = [
        _curve("pop-a", "pop"),
        _curve("pop-b", "pop"),
        _curve("rock-a", "rock"),
        _curve("rock-b", "rock"),
    ]

    panel.knn_k_spin.setValue(1)
    panel.structure_section_spin.setValue(3)
    panel.set_result(matrix, [curve.name for curve in curves], {"labels": [1, 1, 2, 2]}, curves)

    assert panel.genre_table.rowCount() == 2
    assert panel.genre_table.columnCount() == 2
    assert panel.genre_table.item(0, 0).text() == "0.1000"
    assert "Accuracy: 1.0000" in panel.knn_metrics_label.text()
    assert panel.knn_table.rowCount() == 4
    assert panel.confusion_table.item(0, 0).text() == "2"
    assert panel.confusion_table.item(1, 1).text() == "2"
    assert panel.structure_table.rowCount() == 4
    assert panel.structure_table.item(0, 3).text() == "ABA"
    assert panel.method_comparison_table.rowCount() == 4
    assert panel.method_comparison_table.item(3, 0).text() == "DTW"
