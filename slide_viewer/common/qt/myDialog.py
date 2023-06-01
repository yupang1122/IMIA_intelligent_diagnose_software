from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import sys
import time
# from main_un import *
# from ceshi_format import *
# from format_tab import *


class img_viewed(QDialog):

    # def __init__(self,parent = None):
    #     super(img_viewed,self).__init__(parent)
    def __init__(self, slide_path, parent = None):
        super(img_viewed,self).__init__(parent)
        self.slide_path = slide_path
        self.parent = parent
        self.width = 960
        self.height = 500
        self.scroll_ares_images = QScrollArea(self)
        self.scroll_ares_images.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget(self)
        self.scrollAreaWidgetContents.setObjectName('scrollAreaWidgetContends')
        self.gridLayout = QGridLayout(self.scrollAreaWidgetContents)
        self.scroll_ares_images.setWidget(self.scrollAreaWidgetContents)
        self.scroll_ares_images.setGeometry(20, 50, self.width, self.height)
        self.vertocal1 = QVBoxLayout()
        self.vertocal1.addWidget(self.scroll_ares_images)
        self.show()
        #设置图片的预览尺寸；
        self.displayed_image_size = 100
        self.col = 0
        self.row =0
        out_folder = '../dataprocess/slide_tiles/'
        tile_size = 224
        svs_name = self.slide_path.split('/')[-1][:-4]
        out_path = os.path.join(out_folder, svs_name, str(tile_size))  ##变量命名保存位置
        self.tile_path = os.path.join('../dataprocess/ROI_tiles', svs_name)
        if not os.path.exists(self.tile_path):
            os.makedirs(self.tile_path)
        # print(out_path)
        self.initial_path = out_path
        self.start_img_viewer()
        # self.scroll_ares_images.setContextMenuPolicy(Qt.CustomContextMenu)  ##右键开放策略
        # self.scroll_ares_images.customContextMenuRequested.connect(self.right_menu)


    def right_menu(self):
        menu = QMenu(self)
        menu1 = QMenu('图片保存', menu)
        menu2 = QMenu('图片信息', menu)
        menu3 = QMenu('Add TabWidget', menu)

        menu1.addAction(QAction('×2', menu1))
        menu1.addAction(QAction('×5', menu1))
        menu1.addAction(QAction('×10', menu1))
        menu1.addAction(QAction('zoom_in', menu1))
        menu1.addAction(QAction('zoom_out', menu1))
        menu2.addAction(QAction('Rubber Band', menu2))
        menu2.addAction(QAction('Move Tool', menu2))
        menu2.addAction(QAction('Rectangle Tool', menu2))
        menu3.addAction(QAction('Previous File', menu3))
        menu3.addAction(QAction('Current File', menu3))
        menu3.addAction(QAction('Next File', menu3))
        menu.addAction(menu1.menuAction())
        menu.addAction(menu2.menuAction())
        menu.addAction(menu3.menuAction())
        # menu.triggered.connect(self.menuSlot)
        menu.exec_(QCursor.pos())

    def start_img_viewer(self):
        if self.initial_path:
            file_path = self.initial_path
            # print('file_path为{}'.format(file_path))
            # print(file_path)
            img_type = 'jpg'
            if file_path and img_type:

                png_list = list(i for i in os.listdir(file_path) if str(i).endswith('.{}'.format(img_type)))
                # print(png_list)
                num = len(png_list)
                if num !=0:
                    for i in range(num):
                        image_path = str(file_path + '/' + png_list[i])
                        image_id = str(png_list[i])
                        # print(image_id)
                        pixmap = QPixmap(image_path)
                        # pixmap = QImage(image_path)

                        self.addImage(pixmap, image_id)
                        # print(pixmap)
                        QApplication.processEvents()
                else:
                    QMessageBox.warning(self,'错误','生成图片文件为空')
                    self.event(exit())
            else:
                QMessageBox.warning(self,'错误','文件为空，请稍后')
        else:

            QMessageBox.warning(self, '错误', '文件为空，请稍后')



    def loc_fil(self,stre):
        print('存放地址为{}'.format(stre))
        self.initial_path = stre

    def geng_path(self,loc):
        print('路径为，，，，，，{}'.format(loc))
    def gen_type(self,type):
        print('图片类型为：，，，，{}'.format(type))


    def addImage(self, pixmap, image_id):
        #图像法列数
        nr_of_columns = self.get_nr_of_image_columns()
        #这个布局内的数量
        nr_of_widgets = self.gridLayout.count()
        self.max_columns =nr_of_columns
        if self.col < self.max_columns:
            self.col =self.col +1
        else:
            self.col =0
            self.row +=1

        # print('行数为{}'.format(self.row))
        # print('此时布局内不含有的元素数为{}'.format(nr_of_widgets))
        #
        # print('列数为{}'.format(self.col))
        clickable_image = QClickableImage(self.displayed_image_size, self.displayed_image_size, pixmap, image_id, self.tile_path)
        clickable_image.clicked.connect(self.on_left_clicked)
        clickable_image.rightClicked.connect(self.on_right_clicked)
        self.gridLayout.addWidget(clickable_image, self.row, self.col)


    def on_left_clicked(self,image_id):
        print('left clicked - image id = '+image_id)

    def on_right_clicked(self,image_id):
        print('right clicked - image id = ' + image_id)


    def get_nr_of_image_columns(self):
        #展示图片的区域
        scroll_area_images_width = self.width
        if scroll_area_images_width > self.displayed_image_size:

            pic_of_columns = scroll_area_images_width // self.displayed_image_size  #计算出一行几列；
        else:
            pic_of_columns = 1
        return pic_of_columns

    def setDisplayedImageSize(self,image_size):
        self.displayed_image_size =image_size



#
# class QClickableImage(QWidget):
#     image_id =''
#
#     def __init__(self,width =0,height =0,pixmap =None,image_id = ''):
#         QWidget.__init__(self)
#
#         self.layout =QVBoxLayout(self)
#         self.label1 = QLabel()
#         self.label1.setObjectName('label1')
#         self.lable2 =QLabel()
#         self.lable2.setObjectName('label2')
#         self.width =width
#         self.height = height
#         self.pixmap =pixmap
#
#         if self.width and self.height:
#             self.resize(self.width,self.height)
#         if self.pixmap:
#             pixmap = self.pixmap.scaled(QSize(self.width,self.height),Qt.KeepAspectRatio,Qt.SmoothTransformation)
#             self.label1.setPixmap(pixmap)
#             self.label1.setAlignment(Qt.AlignCenter)
#             self.layout.addWidget(self.label1)
#         if image_id:
#             self.image_id =image_id
#             self.lable2.setText(image_id)
#             self.lable2.setAlignment(Qt.AlignCenter)
#             ###让文字自适应大小
#             self.lable2.adjustSize()
#             self.layout.addWidget(self.lable2)
#         self.setLayout(self.layout)
#
#     clicked = pyqtSignal(object)
#     rightClicked = pyqtSignal(object)
#
#     def mouseressevent(self,ev):
#         print('55555555555555555')
#         if ev.button() == Qt.RightButton:
#             print('dasdasd')
#             #鼠标右击
#             self.rightClicked.emit(self.image_id)
#         else:
#             self.clicked.emit(self.image_id)
#
#     def imageId(self):
#         return self.image_id

class QClickableImage(QWidget):
    image_id = ''

    def __init__(self, width=0, height=0, pixmap=None, image_id='', tile_path = None):
        QWidget.__init__(self)

        self.tile_path = tile_path
        self.width = width
        self.height = height
        self.pixmap = pixmap

        self.layout = QVBoxLayout(self)
        self.lable2 = QLabel()
        self.lable2.setObjectName('label2')

        if self.width and self.height:
            self.resize(self.width, self.height)
        if self.pixmap and image_id:
            pixmap = self.pixmap.scaled(QSize(self.width, self.height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label1 = MyLabel(pixmap, image_id,self.tile_path)
            self.label1.setObjectName('label1')
            # self.label1.connect(self.mouseressevent())
            self.layout.addWidget(self.label1)

        if image_id:
            self.image_id = image_id
            self.lable2.setText(image_id.split('\\')[-1])
            self.lable2.setAlignment(Qt.AlignCenter)
            ###让文字自适应大小
            self.lable2.adjustSize()
            self.layout.addWidget(self.lable2)
        self.setLayout(self.layout)

    clicked = pyqtSignal(object)
    rightClicked = pyqtSignal(object)

    def imageId(self):
        return self.image_id


class MyLabel(QLabel):
    global NOP_value, NOP_dict

    def __init__(self, pixmap=None, image_id=None, tile_path = None):
        QLabel.__init__(self)
        self.tile_path = tile_path
        self.pixmap = pixmap
        self.image_id = image_id
        self.setPixmap(pixmap)

        self.setAlignment(Qt.AlignCenter)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.rightMenuShow)  # 开放右键策略

    def rightMenuShow(self, point):
        # 添加右键菜单
        self.popMenu = QMenu()
        ch = QAction(u'切片保存', self)
        sc = QAction(u'切片信息', self)
        # xs = QAction(u'', self)
        self.popMenu.addAction(ch)
        self.popMenu.addAction(sc)
        # self.popMenu.addAction(xs)
        # 绑定事件
        ch.triggered.connect(self.savetile)
        sc.triggered.connect(self.tilemes)
        # xs.triggered.connect(self.rshow)
        self.showContextMenu(QCursor.pos())

    def savetile(self):
        save_path = os.path.join(self.tile_path, self.image_id)
        print(save_path)
        QPixmap.save(self.pixmap, save_path)
        print('save')



    def tilemes(self):
        print(self.image_id)

    def rshow(self):

        '''
        do something
        '''

    def delete(self):
        '''
        do something
        '''

    def reback(self):
        '''
        do something
        '''

    def showContextMenu(self, pos):
        # 调整位置
        '''''
        右键点击时调用的函数
        '''
        # 菜单显示前，将它移动到鼠标点击的位置

        self.popMenu.move(pos)
        self.popMenu.show()

    def menuSlot(self, act):
        print(act.text())