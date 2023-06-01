from PyQt5.QtCore import QRectF, Qt, QEvent, QLine
from PyQt5.QtGui import QPen, QColor, QPainter,  QWheelEvent, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QWidget, QStyleOptionGraphicsItem




class DrawLineGraphicsItem(QGraphicsItem):
    def __init__(self, line:QLine, level, a, b):
        super().__init__()
        print('ab',a,b)
        self.qrectf = QRectF(a, b, 0, 0)
        self.line = line
        self.level = level
        # print('iniline', line,self.line, self.qrectf)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def boundingRect(self):
        return self.qrectf
    #本质上绘制成了矩形区域笔划

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    ):                    ##缩放操作也会调用paint但不会调用主函数
        painter.save()
        pen = QPen(QColor(3, 36, 255, 255))


        pen.setWidth(5)
        painter.setPen(pen)

        # brush = QBrush()
        # brush.setColor(Qt.yellow)     #画刷颜色
        # brush.setStyle(Qt.SolidPattern)  #填充样式
        # painter.setBrush(brush)
        # print('useline', self.line)
        painter.drawLine(self.line)
        print('pen')

         ##画框动作完成后继续进行操作会根据鼠标在框内不断刷新进而实现不同粗细
        ##设置画刷


        painter.restore()
        ##save restore 是针对坐标系统进行的操作
        #save保存当前坐标状态，restore回复上次保存的坐标状态配对操作一个栈对象
        #利用上下不同时运行对线宽进行递减
