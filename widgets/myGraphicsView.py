

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys


class QmyGraphicsView(QGraphicsView):



   mouseMove = pyqtSignal(QPoint)      #鼠标移动
   
   mouseClicked = pyqtSignal(QPoint)   #鼠标单击

   mouseRelease = pyqtSignal(QPoint)

   mouseDoubleClick = pyqtSignal(QPoint)   #鼠标双击

   keyPress = pyqtSignal(QKeyEvent)    #按键按下

   wheelevent = pyqtSignal(QWheelEvent)

   mouseRight = pyqtSignal(QMouseEvent)



    ##========== event 处理函数============
   def mouseMoveEvent(self,event): ##鼠标移动
      point=event.pos()   
      self.mouseMove.emit(point)  #发射信号
      super().mouseMoveEvent(event)

   def mousePressEvent(self,event): ##鼠标单击
      if event.button()==Qt.LeftButton :
         point=event.pos()    
         self.mouseClicked.emit(point) #发射信号
      super().mousePressEvent(event)

   def mouseReleaseEvent(self,event): ##鼠标单击
      if event.button()==Qt.LeftButton :
         point=event.pos()
         self.mouseRelease.emit(point) #发射信号
      super().mouseReleaseEvent(event)

   def mouseDoubleClickEvent(self,event): ##鼠标双击
      if event.button()==Qt.LeftButton :
         point=event.pos()    
         self.mouseDoubleClick.emit(point)    #发射信号
      super().mouseDoubleClickEvent(event)

   def keyPressEvent(self,event):   ##按键按下
      self.keyPress.emit(event)    #发射信号
      super().keyPressEvent(event)

   def mouseRightEvent(self, event):  ##鼠标单击
      if event.button() == Qt.RightButton :
         point = event.pos()
         self.mouseRight.emit(point)  # 发射信号
      super().mouseRightEvent(event)

   def wheelEvent(self, event):  # 滚轮滚动时调用。event是一个QWheelEvent对象
      # angle = event.angleDelta()  # 返回滚轮转过的数值，单位为1/8度.PyQt5.QtCore.QPoint(0, 120)
      # angle = angle / 8  # 除以8之后单位为度。PyQt5.QtCore.QPoint(0, 15)   【向前滚是正数，向后滚是负数  用angle.y()取值】
      # ang = event.pixelDelta()  # 返回滚轮转过的像素值  【不知为何  没有值】
      self.wheelevent.emit(event)  # 发射信号
      super().wheelEvent(event)
      # w = event.pos()  # 返回相对于控件的当前鼠标位置.PyQt5.QtCore.QPoint(260, 173)
