from PyQt5.QtCore import QRectF, Qt, QEvent, QRect,QPoint
from PyQt5.QtGui import QPen, QColor, QPainter,  QWheelEvent, QBrush, QImage,QPolygon
from PyQt5.QtWidgets import QGraphicsItem, QWidget, QStyleOptionGraphicsItem




class SelectedRectGraphicsItem(QGraphicsItem):
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
        pen = QPen(QColor(0, 255, 0, 255))
        if self.level < -0.9:
            self.level = -0.9
        pen.setWidth(self.level + 2)
        # pen.setWidth(2.5)   可进行浮点小数设置
        painter.setPen(pen)
        print('selectrectpaint')
        # brush = QBrush()
        # brush.setColor(Qt.yellow)     #画刷颜色
        # brush.setStyle(Qt.SolidPattern)  #填充样式
        # painter.setBrush(brush)
        # print(self.qrectf,'qrectf')
        painter.drawRect(self.qrectf)
        # print(self.level,'pen')
        self.level -= 0.02
         ##画框动作完成后继续进行操作会根据鼠标在框内不断刷新进而实现不同粗细
        ##设置画刷
        # points = [QPoint(100,100),QPoint(1000,1000),QPoint(500,500)]
        # painter.drawPoints(QPolygon(points))
        # painter.drawPixmap()
        # img = QImage('E:\Code\HistoSlider\dataprocess\ROI_data/area_roi_2021-03-31-17-40-24.jpg')  # 读取图像文件
        # rect = QRect(200, 100, img.width() / 4, img.height() / 4)  # 进行绘制,对图片的大小压说为原来的二分之一
        # painter.drawImage(rect, img)


        painter.restore()
        ##save restore 是针对坐标系统进行的操作
        #save保存当前坐标状态，restore回复上次保存的坐标状态配对操作一个栈对象
        #利用上下不同时运行对线宽进行递减

