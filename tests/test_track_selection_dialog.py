import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.io.midi_loader import MidiTrackInfo
from src.ui.track_selection_dialog import TrackSelectionDialog


def _app():
    """Return a QApplication singleton for widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _track_infos():
    """Return representative track metadata for dialog tests."""
    return [
        MidiTrackInfo(0, "vocal", "Piano", 104, 55, 79, 23.93, 215.56),
        MidiTrackInfo(1, "bass", "Bass", 270, 32, 41, 12.81, 218.43),
    ]


def test_track_selection_dialog_defaults_to_combined_highest():
    _app()
    dialog = TrackSelectionDialog("song.mid", _track_infos())

    assert dialog.selected_options() == (None, "highest")


def test_track_selection_dialog_returns_selected_track_and_mode():
    _app()
    dialog = TrackSelectionDialog("song.mid", _track_infos())

    dialog.track_table.selectRow(1)
    dialog.mode_combo.setCurrentText("全部音符")

    assert dialog.selected_options() == (1, "all")
