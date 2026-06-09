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
    """Create a curve for method-change UI tests."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[Note(timestamp, pitch, velocity) for timestamp, pitch, velocity in notes_data],
    )


def test_main_window_recomputes_when_distance_method_changes(monkeypatch):
    _app()
    window = MainWindow()
    monkeypatch.setattr(window.cluster_panel, "set_result", lambda *args, **kwargs: None)
    window._curves = [
        _curve("a", [(0.0, 60, 80), (1.0, 64, 90), (2.0, 67, 85)]),
        _curve("b", [(0.0, 60, 80), (1.0, 65, 90), (2.0, 69, 85)]),
    ]
    window._known_curve_count = 2

    window._on_compute()
    assert window._matrix_method == "standard"
    standard_distance = window._matrix[0, 1]

    window.control_bar.method_combo.setCurrentText("DTW")

    assert window._matrix_method == "dtw"
    assert window._matrix is not None
    assert window._matrix[0, 1] != standard_distance
