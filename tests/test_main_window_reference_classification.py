import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog

from src.analysis.reference_classifier import ReferenceItem
from src.models.melody_curve import MelodyCurve
from src.ui.main_window import MainWindow


def _app():
    """Return a QApplication singleton for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_classifies_query_midi(monkeypatch):
    _app()
    window = MainWindow()
    references = [
        ReferenceItem("folk-a", "folk", "/fake/folk-a.mid", np.array([[0.0, 0.1, 0.7], [1.0, 0.2, 0.7]])),
        ReferenceItem("rock-a", "rock", "/fake/rock-a.mid", np.array([[0.0, 0.8, 0.7], [1.0, 0.7, 0.7]])),
    ]
    query = MelodyCurve(
        name="query",
        filepath="/fake/query.mid",
        points=np.array([[0.0, 0.11, 0.7], [1.0, 0.21, 0.7]]),
    )

    import src.analysis.reference_classifier as reference_classifier

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("/fake/query.mid", ""))
    monkeypatch.setattr(reference_classifier, "load_or_build_standard_set", lambda: references)
    monkeypatch.setattr(reference_classifier, "load_query_curve", lambda filepath: query)
    window.cluster_panel.knn_k_spin.setValue(1)

    window._on_classify_query()

    assert "预测曲风: folk" in window.control_bar.status_label.text()
    assert "folk-a" in window.control_bar.status_label.text()
    assert "预测曲风: folk" in window.cluster_panel.query_summary_label.text()
    assert window.cluster_panel.query_neighbors_table.item(0, 1).text() == "folk-a"
