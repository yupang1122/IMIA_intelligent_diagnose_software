from PyQt5.QtCore import QRectF, Qt, QEvent
from PyQt5.QtGui import QPen, QColor, QPainter,  QWheelEvent, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QWidget, QStyleOptionGraphicsItem

class PickRectGraphicsItem(QGraphicsItem):
    def __init__(self, qrectf: QRectF, level):
        super().__init__()
        self.qrectf = qrectf
        self.level = level
        # print(self.level)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def boundingRect(self):
        return self.qrectf

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    ):                    ##缩放操作也会调用paint但不会调用主函数
        painter.save()
        pen = QPen(QColor(0, 0, 255, 255))
        if self.level < -0.9:
            self.level = -0.9
        pen.setWidth(self.level + 2)
        # pen.setWidth(2.5)   可进行浮点小数设置
        painter.setPen(pen)
        brush = QBrush()
        brush.setColor(Qt.yellow)     #画刷颜色
        brush.setStyle(Qt.SolidPattern)  #填充样式
        painter.setBrush(brush)
        painter.drawRect(self.qrectf)
        # print(self.level,'pen')
        self.level -= 0.02
         ##画框动作完成后继续进行操作会根据鼠标在框内不断刷新进而实现不同粗细
        ##设置画刷
        painter.restore() ##无法操作复选
        ##save restore 是针对坐标系统进行的操作
        #save保存当前坐标状态，restore回复上次保存的坐标状态配对操作一个栈对象
        #利用上下不同时运行对线宽进行递减