import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.models.melody_curve import MelodyCurve
from src.models.note import Note
from src.ui.main_window import MainWindow


def _app():
    """Return a QApplication singleton for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _curve(name: str, notes_data) -> MelodyCurve:
    """Create a small curve for segment UI tests."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[Note(timestamp, pitch, velocity) for timestamp, pitch, velocity in notes_data],
    )


def test_main_window_reports_best_segment(monkeypatch):
    _app()
    window = MainWindow()
    window._curves = [
        _curve("left", [(0.0, 60, 80), (1.0, 62, 80), (2.0, 64, 80)]),
        _curve("right", [(0.0, 55, 80), (1.0, 60, 80), (2.0, 62, 80), (3.0, 64, 80)]),
    ]
    monkeypatch.setattr(window.file_panel, "get_visible_indices", lambda: [0, 1])
    window.control_bar.segment_window_spin.setValue(0.7)
    window.control_bar.segment_step_spin.setValue(0.3)

    window._on_find_best_segment()

    assert "最相似片段" in window.control_bar.status_label.text()
    assert "距离" in window.control_bar.status_label.text()
