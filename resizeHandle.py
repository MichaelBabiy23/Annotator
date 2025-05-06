#resizeHandle.py
from PySide6.QtWidgets import (
 QGraphicsRectItem, QGraphicsItem,

)
from PySide6.QtGui  import  QPen, QColor
from PySide6.QtCore import  QRectF, QPointF

MIN_BOX_SIZE = 10
HANDLE_SIZE  = 8


# ────────────────────────── Resize handle ────────────────────────────
class ResizeHandle(QGraphicsRectItem):
    def __init__(self, corner: str, parent_box: 'BoxItem'):
        super().__init__(QRectF(0, 0, HANDLE_SIZE, HANDLE_SIZE), parent_box)
        self.corner      = corner
        self.parent_box  = parent_box

        self.setBrush(QColor(255, 255, 255))
        pen = QPen(QColor(0, 0, 0)); pen.setWidth(1)
        self.setPen(pen)

        self.setFlag(QGraphicsItem.ItemIsMovable,            True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(1)

    # ─── keep the parent from moving while we drag ───────────────────
    def mousePressEvent(self, event):
        self._orig_rect = self.parent_box.rect()           # save for undo
        self.parent_box.setFlag(QGraphicsItem.ItemIsMovable, False)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.parent_box.setFlag(QGraphicsItem.ItemIsMovable, True)

        # push undo if the rectangle actually changed
        if hasattr(self, "_orig_rect") and self._orig_rect != self.parent_box.rect():
            box, old_rect = self.parent_box, self._orig_rect
            annotator     = box.scene().views()[0].window()
            def _undo_resize(b=box, r=old_rect):
                b.setRect(r)
                b.update_handles()
            annotator._push_undo(_undo_resize)

    # ─── keep parent rect in sync while dragging the handle ───────────
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            if getattr(self.parent_box, "_updating_handles", False):
                return super().itemChange(change, value)

            # compute a new rectangle for the parent based on this corner
            new_pos    = value
            rect       = self.parent_box.rect()
            global_off = self.parent_box.mapToScene(QPointF(0, 0))

            if self.corner == 'tl':
                x1, y1 = new_pos.x() + global_off.x(), new_pos.y() + global_off.y()
                x2, y2 = rect.right() + global_off.x(), rect.bottom() + global_off.y()
            elif self.corner == 'tr':
                x1, y1 = rect.left() + global_off.x(), new_pos.y() + global_off.y()
                x2, y2 = new_pos.x() + global_off.x(), rect.bottom() + global_off.y()
            elif self.corner == 'bl':
                x1, y1 = new_pos.x() + global_off.x(), rect.top() + global_off.y()
                x2, y2 = rect.right() + global_off.x(), new_pos.y() + global_off.y()
            else:  # 'br'
                x1, y1 = rect.left() + global_off.x(), rect.top() + global_off.y()
                x2, y2 = new_pos.x() + global_off.x(), new_pos.y() + global_off.y()

            scene_rect = QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized()
            local_rect = QRectF(
                self.parent_box.mapFromScene(scene_rect.topLeft()),
                self.parent_box.mapFromScene(scene_rect.bottomRight())
            )

            if (local_rect.width()  < MIN_BOX_SIZE or
                local_rect.height() < MIN_BOX_SIZE):
                return super().itemChange(change, self.pos())  # reject

            self.parent_box.setRect(local_rect)

            # reposition *all* handles once — without recursion
            self.parent_box._updating_handles = True
            self.parent_box.update_handles()
            self.parent_box._updating_handles = False

        return super().itemChange(change, value)