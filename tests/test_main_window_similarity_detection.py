import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog

from src.analysis.melody_similarity import MelodySimilarityResult
from src.analysis.segment_analysis import SegmentMatch
from src.ui.main_window import MainWindow


def _app():
    """Return a QApplication singleton for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_detects_melody_similarity(monkeypatch):
    _app()
    window = MainWindow()
    result = MelodySimilarityResult(
        left_name="left",
        right_name="right",
        modified_distance=0.03,
        dtw_distance=0.04,
        best_segment=SegmentMatch(0.0, 0.25, 0.5, 0.75, 0.02, 8, 9),
        level="高度相似",
        score=81.25,
    )

    import src.analysis.melody_similarity as melody_similarity

    selections = iter([("left.mid", ""), ("right.mid", "")])
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr(melody_similarity, "compare_midi_files", lambda *args, **kwargs: result)

    window._on_detect_similarity()

    assert "旋律检测: 高度相似" in window.control_bar.status_label.text()
    assert "判定: 高度相似" in window.cluster_panel.similarity_summary_label.text()
