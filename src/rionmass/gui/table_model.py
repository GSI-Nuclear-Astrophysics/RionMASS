"""Read-only Qt table model for pandas results."""

from __future__ import annotations

import math

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt


class DataFrameTableModel(QAbstractTableModel):
    """Displays a :class:`pandas.DataFrame` in a ``QTableView``, read-only.

    Floats use 6 significant figures, or the supplied fixed decimal places.
    Unavailable numeric values show an em dash. Purely a display adapter;
    it never mutates ``df``.
    """

    def __init__(self, df: pd.DataFrame | None = None, decimal_places: int | None = None) -> None:
        super().__init__()
        self._df = df if df is not None else pd.DataFrame()
        self._decimal_places = decimal_places

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._df.index)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()) -> int:  # noqa: B008
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return (
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if isinstance(value, (int, float))
                else None
            )
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if isinstance(value, float):
            if math.isnan(value):
                return "—"
            if self._decimal_places is not None:
                return f"{value:.{self._decimal_places}f}"
            return f"{value:.6g}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            label = str(self._df.columns[section])
            return label.replace(" (keV)", "\n(keV)") if self._decimal_places is not None else label
        return str(self._df.index[section])
