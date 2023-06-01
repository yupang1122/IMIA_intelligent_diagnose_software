
from PyQt5.Qt import *


class MyQSlider(QSlider):
    def __init__(self,parent=None,*args,**kwargs):
        super().__init__(parent,*args,**kwargs)
        label = QLabel(self)
        self.label = label
        label.setText('0')
        label.setStyleSheet('background-color:cyan;color:red')
        # label.hide()

    def mousePressEvent(self, evt):
        super().mousePressEvent(evt)
        y = (1-((self.value()-self.minimum())/(self.maximum()-self.minimum())))*(self.height()-self.label.height())
        x =  (self.width()-self.label.width())/2
        # self.label.move(y,x)
        self.label.show()
        self.label.setText(str(self.value()))

    def mouseMoveEvent(self, evt):
        super().mouseMoveEvent(evt)
        y = (1-((self.value()-self.minimum())/(self.maximum()-self.minimum())))*(self.height()-self.label.height())
        x =  (self.width()-self.label.width())/2
        # self.label.move(y,x)
        self.label.setText(str(self.value()))
        self.label.adjustSize()

    def mouseReleaseEvent(self, evt):
        super().mouseReleaseEvent(evt)
        # self.label.hide()

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('QSlider_案例')
        self.resize(500, 500)
        self.mainProject()

    def mainProject(self):

        sd = MyQSlider(self)
        self.sd = sd
        sd.setMaximum(200)
        sd.setMinimum(100)
        sd.resize(self.width() /12, self.height() *3/ 5)
        sd.move((self.width() - sd.width()) / 2, (self.height() - sd.height()) * 0.2 / 2)
        sd.setSingleStep(2)  # 设置步长
        sd.setPageStep(5)  # 设置翻页步长，使用PageUp PageDown
        sd.setTracking(True)
        # sd.setValue(101)  # 设置数值
        # sd.setSliderPosition(199)  # 设置滑块位置
        # sd.setInvertedAppearance(True)  # 反转外观
        # sd.setInvertedControls(True)  # 反转操作，
        sd.setOrientation(Qt.Vertical)
        self.sd.setTickPosition(QSlider.TicksBothSides)
        self.sd.setTickInterval(5)
        self.sd.setPageStep(5)  # 设置翻页步长，也会顺带调整刻度线密度

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    win = MyWindow()
    win.show()
    sys.exit(app.exec_())
