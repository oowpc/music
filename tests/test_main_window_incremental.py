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


def _curve(name: str, offset: int) -> MelodyCurve:
    """Create a small curve for main-window incremental tests."""
    return MelodyCurve(
        name=name,
        filepath=f"/fake/{name}.mid",
        raw_notes=[
            Note(0.0, 60 + offset, 80),
            Note(1.0, 64 + offset, 90),
        ],
    )


def test_main_window_extends_matrix_when_curve_is_appended(monkeypatch):
    _app()
    window = MainWindow()
    monkeypatch.setattr(window.cluster_panel, "set_result", lambda *args, **kwargs: None)
    curves = [_curve("a", 0), _curve("b", 2)]
    window._curves = list(curves)
    window._known_curve_count = 2
    window._on_compute()

    calls = []

    def spy_extend_matrix(existing_matrix, all_curves, previous_count, method):
        from src.analysis.distance_matrix import build_matrix

        calls.append((existing_matrix.shape, len(all_curves), previous_count, method))
        return build_matrix(all_curves, method=method)

    import src.analysis.distance_matrix as distance_matrix

    monkeypatch.setattr(distance_matrix, "extend_matrix", spy_extend_matrix)

    curves.append(_curve("c", 4))
    monkeypatch.setattr(window.file_panel, "get_curves", lambda: curves)

    window._on_files_loaded()

    assert calls == [((2, 2), 3, 2, "standard")]
    assert window._matrix.shape == (3, 3)
