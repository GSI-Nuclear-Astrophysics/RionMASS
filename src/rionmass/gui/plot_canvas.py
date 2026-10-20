"""Matplotlib canvas and navigation toolbar for Qt."""

from __future__ import annotations

from collections.abc import Callable

from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotCanvas(QWidget):
    """Embeds one Matplotlib :class:`~matplotlib.axes.Axes` per call to :meth:`draw_with`."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def draw_with(self, plot_fn: Callable[..., object], *args, **kwargs) -> None:
        """Clear the figure, call ``plot_fn(*args, ax=ax, **kwargs)``, and redraw."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plot_fn(*args, ax=ax, **kwargs)
        self.figure.tight_layout()
        self.canvas.draw_idle()
