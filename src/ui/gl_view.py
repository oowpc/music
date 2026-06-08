import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


try:
    import pyqtgraph.opengl as gl

    HAS_PYQTGRAPH = True
except Exception:
    HAS_PYQTGRAPH = False
    gl = None


class GLView(QWidget):
    """3D melody curve view backed by PyQtGraph OpenGL when available."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._line_items = []

        if HAS_PYQTGRAPH:
            self._gl_widget = gl.GLViewWidget()
            self._gl_widget.setCameraPosition(distance=2.5, elevation=30, azimuth=-45)

            grid = gl.GLGridItem()
            grid.setSize(2, 2, 2)
            grid.setSpacing(0.25, 0.25)
            grid.translate(0.5, 0.5, 0)
            self._gl_widget.addItem(grid)

            layout.addWidget(self._gl_widget)
        else:
            self._gl_widget = None
            label = QLabel("pyqtgraph 未安装，3D 视图不可用")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

    def set_curves(self, curves) -> None:
        """Render curves whose points arrays have shape (N, 3)."""
        if self._gl_widget is None:
            return

        for item in self._line_items:
            self._gl_widget.removeItem(item)
        self._line_items.clear()

        for curve in curves:
            if curve.points is None or len(curve.points) == 0:
                continue

            line_data = np.asarray(curve.points, dtype=np.float32)
            red, green, blue = _hex_to_rgb(curve.color)
            line = gl.GLLinePlotItem(
                pos=line_data,
                color=(red, green, blue, 1.0),
                width=2.0,
                antialias=True,
            )
            self._gl_widget.addItem(line)
            self._line_items.append(line)


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    """Convert #rrggbb color text to normalized RGB floats."""
    clean = color.lstrip("#")
    if len(clean) != 6:
        clean = "ffffff"
    return (
        int(clean[0:2], 16) / 255.0,
        int(clean[2:4], 16) / 255.0,
        int(clean[4:6], 16) / 255.0,
    )
