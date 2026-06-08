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
    """Create a curve for weight UI tests."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[Note(timestamp, pitch, velocity) for timestamp, pitch, velocity in notes_data],
    )


def test_main_window_weights_change_distance_and_invalidate(monkeypatch):
    _app()
    window = MainWindow()
    monkeypatch.setattr(window.cluster_panel, "set_result", lambda *args, **kwargs: None)
    window._curves = [
        _curve("a", [(0.0, 60, 80), (1.0, 62, 80)]),
        _curve("b", [(0.0, 72, 80), (1.0, 74, 80)]),
    ]
    window._known_curve_count = 2

    window._on_compute()
    default_distance = window._matrix[0, 1]

    window.control_bar.pitch_weight_spin.setValue(2.0)
    assert window._matrix is None

    window._on_compute()
    weighted_distance = window._matrix[0, 1]

    assert weighted_distance > default_distance


def test_control_bar_returns_dimension_weights():
    _app()
    window = MainWindow()

    window.control_bar.time_weight_spin.setValue(0.5)
    window.control_bar.pitch_weight_spin.setValue(2.0)
    window.control_bar.velocity_weight_spin.setValue(0.25)

    assert window.control_bar.get_dimension_weights() == (0.5, 2.0, 0.25)
