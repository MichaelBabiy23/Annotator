# GraphicsView.py
from PySide6.QtWidgets import QGraphicsView
from PySide6.QtCore    import Qt, QPoint


# ───────────────────────────── Graphics view ─────────────────────────
class GraphicsView(QGraphicsView):
    """Zoom with Ctrl+wheel, fit‑to‑view on double‑click, and
    **right‑button drag to pan** around the scene.
    """
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints())
        self.setMouseTracking(True)

        # panning state
        self._panning   = False
        self._pan_start = QPoint()

    # ─────────────── zoom with mouse wheel ────────────────
    def wheelEvent(self, ev):
        """Plain wheel → zoom (no Ctrl/Shift/Alt required)."""
        factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        ev.accept()

    # ─────────────── fit‑to‑view (double‑click) ─────────────────────
    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
        else:
            super().mouseDoubleClickEvent(ev)

    # ─────────────── panning with right‑drag ────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._panning   = True
            self._pan_start = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning:
            delta = ev.pos() - self._pan_start
            self._pan_start = ev.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value()   - delta.y())
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton and self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)
