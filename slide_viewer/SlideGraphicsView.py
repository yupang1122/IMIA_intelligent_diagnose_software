from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView, QApplication


class SlideGraphicsView(QGraphicsView):
    def __init__(self, scene, *args, zoomhandler=None, **kwargs):
        super().__init__(scene, *args, **kwargs)
        self.zoom_handler = None

    def setZoomhandler(self, zoomhandler):
        self.zoom_handler = zoomhandler

    def get_current_scene_window(self):
        size = self.size()
        points = self.mapToScene(0, 0, size.width(), size.height()).boundingRect()
        (x, y, w, h) = (points.x(), points.y(), points.width(), points.height())
        return x, y, w, h

    def updateSlideView(self):
        (x, y, w, h) = self.get_current_scene_window()
        self.scene().paint_view(self, x, y, w, h, self.scene().cur_downsample)

    def paintEvent(self, event):
        self.updateSlideView()
        super().paintEvent(event)

    def mouseDoubleClickEvent(self, event):
        point = self.mapToScene(event.pos())
        x = point.x()
        y = point.y()
        print('double')
        self.centerOn(x, y)
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        point = self.mapToScene(event.pos())
        x = point.x()
        y = point.y()
        event.ignore()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  ##鼠标移动
        point = self.mapToScene(event.pos())
        x = point.x()
        y = point.y()
        event.ignore()
        # print('move')
        self.updateSlideView()
        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event):  ##鼠标单击
        point = self.mapToScene(event.pos())
        x = point.x()
        y = point.y()
        event.ignore()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event, *args, **kwargs):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            super().wheelEvent(event, *args, **kwargs)
        else:
            if self.zoom_handler is not None:
                curVal = self.zoom_handler.value()
                numDegrees = event.delta() / 8
                numSteps = numDegrees / 15
                if numSteps > 0:
                    zoom_val = max(curVal - curVal * 0.1 * numSteps, 0.001)
                else:
                    zoom_val = curVal - curVal * 0.1 * numSteps
                self.zoom_handler.setValue(zoom_val)
