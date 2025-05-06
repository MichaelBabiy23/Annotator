# box_item.py
from resizeHandle import ResizeHandle, MIN_BOX_SIZE, HANDLE_SIZE
import sys, os, json
from typing import List
from PySide6.QtWidgets import (QGraphicsRectItem, QGraphicsItem)
from PySide6.QtGui  import QPen, QColor
from PySide6.QtCore import Qt, QRectF, QPointF



# ───────────────────────────── Box item ──────────────────────────────
class BoxItem(QGraphicsRectItem):
    # Define class colors
    CLASS_COLORS = {
        0: QColor(0, 255, 0),    # Class 0 (bottle): Green
        1: QColor(255, 0, 0),    # Class 1: Red
        2: QColor(0, 0, 255),    # Class 2: Blue
        3: QColor(255, 255, 0),  # Class 3: Yellow
        4: QColor(255, 0, 255),  # Class 4: Magenta
        5: QColor(0, 255, 255),  # Class 5: Cyan
        6: QColor(255, 128, 0),  # Class 6: Orange
        7: QColor(128, 0, 255),  # Class 7: Purple
        8: QColor(0, 128, 255),  # Class 8: Light Blue
        9: QColor(255, 128, 128) # Class 9: Pink
    }
    
    def __init__(self, rect: QRectF, cls=0, conf=1.0):
        super().__init__(rect)
        self.cls, self.conf = cls, conf
        self._update_pen()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable,     True)
        self.handles: List[ResizeHandle] = []
        self._updating_handles = False

    def _update_pen(self):
        """Update the pen color based on the class."""
        # Get color for class, default to green if not in the color map
        color = self.CLASS_COLORS.get(self.cls, QColor(0, 255, 0))
        pen = QPen(color)
        pen.setWidth(2)
        self.setPen(pen)

    def set_class(self, cls: int):
        """Set the class and update the pen color."""
        self.cls = cls
        self._update_pen()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._orig_pos = self.pos()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if hasattr(self, "_orig_pos") and self.pos() != self._orig_pos:
            box, old_pos = self, self._orig_pos
            annot = box.scene().views()[0].window()
            annot._push_undo(lambda b=box, p=old_pos: b.setPos(p))

    def _corner_positions(self):
        r = self.rect()
        return {
            'tl': QPointF(r.left(),                r.top()),
            'tr': QPointF(r.right() - HANDLE_SIZE, r.top()),
            'bl': QPointF(r.left(),                r.bottom() - HANDLE_SIZE),
            'br': QPointF(r.right() - HANDLE_SIZE, r.bottom() - HANDLE_SIZE)
        }

    def add_handles(self):
        if self.handles:
            return
        for c in ('tl', 'tr', 'bl', 'br'):
            self.handles.append(ResizeHandle(c, self))
        self.update_handles()

    def remove_handles(self):
        """Detach & forget every resize handle, ignoring ones already gone."""
        for h in self.handles[:]:  # iterate over a copy
            try:
                sc = h.scene()  # may raise if C++ object is gone
            except RuntimeError:
                sc = None
            if sc is not None:
                sc.removeItem(h)
        self.handles.clear()

    def update_handles(self):
        for h in self.handles:
            h.setPos(self._corner_positions()[h.corner])