import os
import imghdr
import openslide
import psutil
import numpy as np
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlRecord
from PyQt5.QtWebEngineWidgets import *
from slide_viewer.common.qt.myDelegates import QmyComboBoxDelegate
from enum import Enum  ##枚举类型
from slide_viewer.common.SlideViewParams import SlideViewParams
from resources.ui.demo13 import Ui_MainWindow
from widgets.SlideViewerWidget import SlideViewerWidget
from widgets.myGraphicsView import QmyGraphicsView
from slide_viewer.common.qt.mySlider import MyQSlider
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  #大尺寸像素多的图片拉高物理内存占用阈值致使不再报错

class TreeItemType(Enum):  ##节点类型枚举类型
    itTopItem = 1001  # 顶层节点
    itGroupItem = 1002  # 组节点
    itImageItem = 1003  # 图片文件节点

class TreeColNum(Enum):  ##目录树的列号枚举类型
    colItem = 0  # 分组/文件名列
    colItemType = 1  # 节点类型列

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.memory_init()
        self.viewer = SlideViewerWidget()  ##使用tabclean清除
        self.thuPixmap = QPixmap()
        self.init_Window()
        self.build_StatusBar()  # 构造状态栏
        self.iniThuArea()
        self.QSlider_init()
        self.Legend_init()
        self.timer = QBasicTimer()
        self.barstep = 0
        self.exist_file = False
        print('init')

    def memory_init(self):
        self.process = psutil.Process(os.getpid())  # 内存管理
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.update_memory_usage)
        # 通过设置计数器刷新内存占用显示 实时监控内存占用情况
        self.ui_timer.start(1000)  ##初始化

    def QSlider_init(self):
        ##滑动条初始化
        self.AlphaSlider = MyQSlider(self.groupBoxFilter_5)
        self.AlphaSlider.resize(self.width() /12, self.height() *3/ 5)
        self.AlphaSlider.move((self.width() - self.AlphaSlider.width()) / 2, (self.height() - self.AlphaSlider.height()) * 0.2 / 2)
        self.AlphaSlider.setMaximum(255)
        self.AlphaSlider.setMinimum(0)
        self.AlphaSlider.setValue(0)  # 设置数值
        self.AlphaSlider.setOrientation(Qt.Horizontal)
        self.AlphaSlider.setTickPosition(QSlider.TicksBelow)
        self.AlphaSlider.setSingleStep(10)  #
        self.AlphaSlider.setPageStep(20)  # 设置翻页步长，也会顺带调整刻度线密度
        self.gridLayout_8.addWidget(self.AlphaSlider, 0, 0, 1, 1)
        self.CancerRateSlider = MyQSlider(self.groupBoxFilter_4)
        self.CancerRateSlider.setMaximum(255)
        self.CancerRateSlider.setMinimum(0)
        self.CancerRateSlider.setSingleStep(10)
        self.CancerRateSlider.setPageStep(20)
        self.CancerRateSlider.setOrientation(QtCore.Qt.Horizontal)
        self.CancerRateSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.CancerRateSlider.setObjectName("CancerRateSlider")
        self.gridLayout_7.addWidget(self.CancerRateSlider, 0, 0, 1, 1)

    def Legend_init(self):
        main_layout = QVBoxLayout()
        a_label = QLabel('脂肪')
        a_label.setAlignment(Qt.AlignCenter)
        a_label.setStyleSheet("background-color:hsv(239, 244, 255)")

        b_label = QLabel('背景')
        b_label.setAlignment(Qt.AlignCenter)
        b_label.setStyleSheet("background-color:hsv(303, 129, 255)")

        c_label = QLabel('碎片')
        c_label.setAlignment(Qt.AlignCenter)
        c_label.setStyleSheet("background-color:hsv(272, 252, 255)")

        d_label = QLabel('淋巴')
        d_label.setAlignment(Qt.AlignCenter)
        d_label.setStyleSheet("background-color:hsv(217, 106, 255)")

        e_label = QLabel('肌肉')
        e_label.setAlignment(Qt.AlignCenter)
        e_label.setStyleSheet("background-color:hsv(168, 223, 255)")

        f_label = QLabel('粘液')
        f_label.setAlignment(Qt.AlignCenter)
        f_label.setStyleSheet("background-color:hsv(120, 253, 255)")

        g_label = QLabel('正常')
        g_label.setAlignment(Qt.AlignCenter)
        g_label.setStyleSheet("background-color:hsv(50, 211, 255)")

        h_label = QLabel('基质')
        h_label.setAlignment(Qt.AlignCenter)
        h_label.setStyleSheet("background-color:hsv(32, 248, 255)")

        i_label = QLabel('肿瘤')
        i_label.setAlignment(Qt.AlignCenter)
        i_label.setStyleSheet("background-color:hsv(359, 253, 255)")

        #图注部分的初始化
        parameter_layout = QVBoxLayout()
        parameter_layout.addWidget(a_label)
        parameter_layout.addWidget(b_label)
        parameter_layout.addWidget(c_label)
        parameter_layout.addWidget(d_label)
        parameter_layout.addWidget(e_label)
        parameter_layout.addWidget(f_label)
        parameter_layout.addWidget(g_label)
        parameter_layout.addWidget(h_label)
        parameter_layout.addWidget(i_label)
        main_layout.addLayout(parameter_layout)

        ##全部添加到mainlayout
        self.Legendlabel.setLayout(main_layout)
        print('legendinit')

    def right_menu(self):
        menu = QMenu(self)
        menu1 = QMenu('Displaywidget', menu)
        menu2 = QMenu('Set Tool', menu)
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
        menu.triggered.connect(self.menuSlot)
        menu.exec_(QCursor.pos())

    def menuSlot(self, act):

        if act.text() == 'Previous File':
            viewer = SlideViewerWidget()
            slide_path = self.viewer.slide_viewer.slide_helper.slide_path
            self.scrollArea.raise_()  ##为了增加上页新建一个viewer
            viewer.slide_viewer.load(SlideViewParams(slide_path))
            self.tabWidget.addTab(viewer, '55')
            return True
        elif act.text() == 'Current File':
            print('currrent')
            return True
        elif act.text() == 'Next File':
            print('next')
            return True
        else:
            self.viewer.slide_viewer.menuSlot(act)

    def build_StatusBar(self):  ##构造状态栏

        self.__labViewCord = QLabel("View：")
        self.__labViewCord.setMinimumWidth(60)
        self.statusBar.addWidget(self.__labViewCord)

        self.__labSceneCord = QLabel("Scene：")
        self.__labSceneCord.setMinimumWidth(60)
        self.statusBar.addWidget(self.__labSceneCord)

        self.__labItemCord = QLabel("Item：")
        self.__labItemCord.setMinimumWidth(60)
        self.statusBar.addWidget(self.__labItemCord)

        self.__labZoomNum = QLabel("Zoom：")
        self.__labZoomNum.setMinimumWidth(60)
        self.statusBar.addWidget(self.__labZoomNum)

        # 缩略图状态栏
        self.__thuViewCord = QLabel("tv：")
        self.__thuViewCord.setMinimumWidth(60)
        self.statusBar.addWidget(self.__thuViewCord)

        self.__thuSceneCord = QLabel("ts：")
        self.__thuSceneCord.setMinimumWidth(60)
        self.statusBar.addWidget(self.__thuSceneCord)

        self.__thuItemCord = QLabel("thu：")
        self.__thuItemCord.setMinimumWidth(60)
        self.statusBar.addWidget(self.__thuItemCord)

        self.memory_usage_label = QLabel()
        self.memory_usage_label.setMinimumWidth(60)
        self.statusBar.addPermanentWidget(self.memory_usage_label)

        self.time_use_bar = QProgressBar()
        self.time_use_bar.setValue(0)
        self.time_use_bar.setMinimum(0)
        self.time_use_bar.setMaximum(100)
        self.statusBar.addWidget(self.time_use_bar)

    def init_Window(self):  ##初始化目录树
        self.itemFlags = (Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                          | Qt.ItemIsEnabled | Qt.ItemIsAutoTristate)  # 节点标志初始化
        self.treeFiles.clear()
        icon = QIcon("./resources/icons/15.ico")

        item = QTreeWidgetItem(TreeItemType.itTopItem.value)
        item.setIcon(TreeColNum.colItem.value, icon)
        item.setText(TreeColNum.colItem.value, "图片文件")
        item.setFlags(self.itemFlags)
        item.setCheckState(TreeColNum.colItem.value, Qt.Checked)

        item.setData(TreeColNum.colItem.value, Qt.UserRole, "")
        self.treeFiles.addTopLevelItem(item)

        self.model = QDirModel()  # 当前的系统model
        self.treeViewOverview.setModel(self.model)
        self.treeViewOverview.show()
        #初始化封存按键防止误触
        self.BackFilterShow.setEnabled(False)
        self.TilesShow.setEnabled(False)
        self.Table_Show.setEnabled(False)
        self.act_database.setEnabled(False)
        self.circle_roi.setEnabled(False)
        self.actThu.setEnabled(False)
        self.viewParams.setEnabled(True)
        self.actZoomIn.setEnabled(False)
        self.actZoomOut.setEnabled(False)  # 功能初始化
        self.act_rubberband.setEnabled(False)
        # self.act_rubberband.setCheckable(False)
        self.act_Location.setEnabled(False)
        self.act_screenshot.setEnabled(False)
        self.act_reset.setEnabled(False)
        self.itenParams.setEnabled(False)
        self.click_tiles.setEnabled(False)
        self.set_grid.setEnabled(False)
        self.show_grid.setEnabled(False)
        self.background.setEnabled(False)
        self.pickgrid.setEnabled(False)
        self.ResultShow.setEnabled(False)
        self.actchoosecancertype.setEnabled(False)
        self.roi_save.setEnabled(False)
        self.roi_tile_save.setEnabled(False)
        self.actionExit.triggered.connect(lambda: QApplication.exit())   #退出功能

        self.tableView.setSelectionBehavior(QAbstractItemView.SelectItems)  ##数据库列表设计
        self.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tableView.setAlternatingRowColors(True)
        self.tableView.verticalHeader().setDefaultSectionSize(22)
        self.tableView.horizontalHeader().setDefaultSectionSize(60)

        self.scrollArea.setContextMenuPolicy(Qt.CustomContextMenu)  ##右键开放策略
        self.scrollArea.customContextMenuRequested.connect(self.right_menu)

    def iniThuArea(self):
        self.thuview = QmyGraphicsView(self)
        self.horizontalLayout_4.addWidget(self.thuview)
        TH = self.thuview.height()
        TW = self.thuview.width()
        Trect = QRectF(-TW / 2, -TH / 2, TW, TH)
        self.thuscene = QGraphicsScene(Trect)
        self.thuview.setScene(self.thuscene)  # 与缩略图关联
        # self.thuview.setCursor(Qt.CrossCursor)  # 设置view鼠标形态
        self.thuview.setMouseTracking(True)
        self.thuview.setDragMode(QGraphicsView.RubberBandDrag)
        ##  4个信号与槽函数的关联
        self.thuview.mouseRelease.connect(self.thu_mouseRelease)
        self.thuview.mouseMove.connect(self.thu_mouseMove)  # 鼠标移动

    def __changeItemCaption(self, item):  ##递归调用函数，修改节点标题
        title = "*" + item.text(TreeColNum.colItem.value)
        item.setText(TreeColNum.colItem.value, title)
        if (item.childCount() > 0):
            for i in range(item.childCount()):
                self.__changeItemCaption(item.child(i))

    @pyqtSlot()  ##添加目录节点
    def on_actTree_AddFolder_triggered(self):
        dirStr = QFileDialog.getExistingDirectory()  # 选择目录
        if (dirStr == ""):
            return

        parItem = self.treeFiles.currentItem()  # 当前节点
        if (parItem == None):
            parItem = self.treeFiles.topLevelItem(0)

        icon = QIcon("./resources/icons/open3.bmp")

        dirObj = QDir(dirStr)  # QDir对象
        nodeText = dirObj.dirName()  # 最后一级目录的名称

        item = QTreeWidgetItem(TreeItemType.itGroupItem.value)  # 节点类型
        item.setIcon(TreeColNum.colItem.value, icon)
        item.setText(TreeColNum.colItem.value, nodeText)  # 第1列
        item.setText(TreeColNum.colItemType.value, "Group")  # 第2列
        item.setFlags(self.itemFlags)
        item.setCheckState(TreeColNum.colItem.value, Qt.Checked)

        item.setData(TreeColNum.colItem.value, Qt.UserRole, dirStr)  # 关联数据为目录全名
        parItem.addChild(item)
        parItem.setExpanded(True)  # 展开节点

    @pyqtSlot()  ##添加图片文件节点          信号与槽装饰器
    def on_actTree_AddFiles_triggered(self):
        options = QFileDialog.Options()  ##打开文件选框
        file_ext_strings = ["*" + ext for ext in self.available_formats]
        file_ext_string = " ".join(file_ext_strings)
        file_path, flt = QFileDialog.getOpenFileNames(
            self,
            "Select whole-slide image to view",
            "../test-data",  ##双点指定项目最高级目录
            "Whole-slide images ({});;".format(file_ext_string),  ##为括号指定后面的格式
            options=options,
        )
        # file_path, flt = QFileDialog.getOpenFileNames(self,  "选择一个或多个文件", "", "Images(*.jpg *.svs *.png *.tiff *.tif *.gif)")
        # 多选文件,返回两个结果，fileList是一个列表类型，存储了所有文件名； flt是设置的文件filter，即"Images(*.jpg)"
        if (len(file_path) < 1):  # fileList是list[str]
            return
        item = self.treeFiles.currentItem()  # 当前节点

        parItem = self.treeFiles.currentItem()  # 当前节点
        if (parItem == None):
            parItem = self.treeFiles.topLevelItem(0)

        if (item == None):
            item = self.treeFiles.topLevelItem(0)
        elif (item.type() == TreeItemType.itImageItem.value):  # 若当前节点是图片节点，取其父节点作为父节点
            parItem = item.parent()
        elif (item.type() == TreeItemType.itGroupItem.value):  # 否则取当前节点为父节点
            parItem = item
        elif (item.type() == TreeItemType.itTopItem.value):
            parItem = item
        icon = QIcon("./resources/icons/31.ico")
        for i in range(len(file_path)):
            fullFileName = file_path[i]  # 带路径文件名
            fileinfo = QFileInfo(fullFileName)
            nodeText = fileinfo.fileName()  # 不带路径文件名
            item = QTreeWidgetItem(TreeItemType.itImageItem.value)  # 节点类型
            item.setIcon(TreeColNum.colItem.value, icon)  # 第1列的图标
            item.setText(TreeColNum.colItem.value, nodeText)  # 第1列的文字
            item.setText(TreeColNum.colItemType.value, "Image")  # 第2列的文字
            item.setFlags(self.itemFlags)
            item.setCheckState(TreeColNum.colItem.value, Qt.Checked)
            item.setData(TreeColNum.colItem.value, Qt.UserRole, fullFileName)  # 关联数据为文件全名
            parItem.addChild(item)

        parItem.setExpanded(True)  # 展开节点

    @pyqtSlot()  ##删除当前节点
    def on_actTree_DeleteItem_triggered(self):
        item = self.treeFiles.currentItem()
        parItem = item.parent()
        parItem.removeChild(item)

    @pyqtSlot()  ##遍历节点
    def on_actTree_ScanItems_triggered(self):
        count = self.treeFiles.topLevelItemCount()
        for i in range(count):
            item = self.treeFiles.topLevelItem(i)
            self.__changeItemCaption(item)

    def on_treeFiles_currentItemChanged(self, current, previous):
        if (current == None):
            return
        nodeType = current.type()  # 获取节点类型
        if (nodeType == TreeItemType.itTopItem.value):  # 顶层节点
            self.actTree_AddFolder.setEnabled(True)
            self.actTree_AddFiles.setEnabled(True)
            self.actTree_DeleteItem.setEnabled(False)  # 顶层节点不能删除

        elif (nodeType == TreeItemType.itGroupItem.value):  # 组节点
            self.actTree_AddFolder.setEnabled(True)
            self.actTree_AddFiles.setEnabled(True)
            self.actTree_DeleteItem.setEnabled(True)

        elif (nodeType == TreeItemType.itImageItem.value):  # 图片节点
            self.actTree_AddFolder.setEnabled(False)
            self.actTree_AddFiles.setEnabled(True)
            self.actTree_DeleteItem.setEnabled(True)
            filename = current.data(TreeColNum.colItem.value, Qt.UserRole)
            # self.viewer = SlideViewerWidget()  ##在此处初始化比较好
            self.tabWidget.clear()
            self.scrollArea.raise_()
            self.file_path = filename
            self.load_slide(filename)
            self.load_thumb(filename)

    def update_memory_usage(self): # return the memory usage in MB
        self.mem = self.process.memory_info()[0] / float(2 ** 20)
        self.memory_usage_label.setText(f"内存占用: {self.mem:.2f} Mb")

    def load_thumb(self, file_path):
        TW = self.thuview.width()
        TH = self.thuview.height()
        imgType_list = self.whole_slide_format
        if imghdr.what(file_path) in imgType_list:
            slide = openslide.OpenSlide(file_path)  # 读入图片（）
            slide_thumbnail = slide.get_thumbnail((4 * TW, 4 * TH))  # 设置成四倍保证清晰度
            size = (4 * TW, 4 * TH)
            self.dimensions = slide.level_dimensions[0]
            downsample = max(*[dim / thumb for dim, thumb in
                               zip(self.dimensions, size)])
            self.thu_level = slide.get_best_level_for_downsample(downsample)
            self.thu_level_dimensions = slide.level_dimensions[self.thu_level]
            thuregion = np.array(slide_thumbnail)

        else:
            thuimg = cv2.imread(file_path)  # 原始图片
            thuresize = cv2.resize(thuimg, (4 * TW, 4 * TH))
            r, g, b = cv2.split(thuresize)
            thuregion = cv2.merge([b, g, r])
            self.thu_level = 0
        thudata = Image.fromarray(thuregion)
        self.thuPixmap = thudata.toqpixmap()
        if self.thuPixmap.width() > self.thuPixmap.height():
            self.thupix = self.thuPixmap.scaledToWidth(TW)
        else:
            self.thupix = self.thuPixmap.scaledToHeight(TH)
        self.thuscene.clear()  # 删除绘图项
        thuitem = QGraphicsPixmapItem(self.thupix)
        thuitem.setPos(-TW / 2, -TH / 2)
        self.thuscene.addItem(thuitem)
        self.thuscene.clearSelection()
        thuitem.setSelected(True)

    def load_slide(self, file_path):
        # self.viewer = SlideViewerWidget()   ##在此处初始化会导致复原操作产生多页面，
        # 为防止内存溢出尽量不使用多页面 采取随时擦除的策略
        self.viewer.slide_viewer.load(SlideViewParams(file_path))
        file_name = os.path.basename(file_path) #获得文件名
        print('treefile', file_path, file_name)

        out_folder = '../dataprocess/slide_tiles/'

        svs_name = file_path.split('/')[-1][:-4]
        out_path = os.path.join(out_folder, svs_name)  ##变量命名保存位置


        out_path = out_path.replace('\\', '/')

        self.time_use_bar.setValue(0)
        self.barstep = 0
        self.timer.stop()
        if os.path.exists(out_path):
            self.exist_file = True
            self.actchoosecancertype.setEnabled(True)
            self.actchoosecancertype.setCheckable(False)
            self.background.setEnabled(True)
            self.background.setCheckable(False)
            self.BackFilterShow.setEnabled(True)
            self.BackFilterShow.setCheckable(False)
            self.roi_tile_save.setEnabled(True)
            self.TilesShow.setEnabled(True)
            self.Table_Show.setEnabled(True)
            # self.ResultShow.setEnabled(True)
        else:
            self.exist_file = False
            self.actchoosecancertype.setEnabled(False)
            self.background.setEnabled(False)
            self.BackFilterShow.setEnabled(False)
            self.roi_tile_save.setEnabled(False)
            self.TilesShow.setEnabled(False)
            self.Table_Show.setEnabled(False)
            self.ResultShow.setEnabled(False)

        self.tabWidget.addTab(self.viewer, file_name)
        self.actZoomIn.setEnabled(True)
        self.actZoomOut.setEnabled(True)
        self.act_rubberband.setEnabled(True)  # 功能初始化
        self.act_rubberband.setCheckable(True)
        self.act_database.setEnabled(True)
        self.circle_roi.setEnabled(True)
        self.actThu.setEnabled(True)
        # self.roi_save.setEnabled(True)
        self.act_Location.setEnabled(True)
        self.act_screenshot.setEnabled(True)
        self.act_reset.setEnabled(True)
        self.itenParams.setEnabled(True)
        self.viewParams.setEnabled(True)
        self.set_grid.setEnabled(True)
        # self.set_grid.setCheckable(True)
        self.show_grid.setEnabled(True)
        self.show_grid.setCheckable(True)

        self.pickgrid.setEnabled(True)
        self.pickgrid.setCheckable(True)
        self.click_tiles.setEnabled(True)
        self.click_tiles.setCheckable(True)



        self.AlphaSlider.valueChanged.connect(self.alpha_slider_valuechange)
        self.CancerRateSlider.valueChanged.connect(self.rate_slider_valuechange)

        QPixmapCache.clear()

    def alpha_slider_valuechange(self, value):
        print(self.AlphaSlider.value())
        clarity = self.AlphaSlider.value()
        if self.viewer.slide_viewer.slide_view_params.grid_visible == True:
            self.viewer.set_alpha(clarity)
        else:
            return True

    def rate_slider_valuechange(self, value):
        print(self.CancerRateSlider.value())
        cancer_rate = self.CancerRateSlider.value()
        if self.viewer.slide_viewer.slide_view_params.grid_visible == True:
            self.viewer.set_rate(cancer_rate)
        else:
            return True

    def thu_mouseMove(self, point):  ##鼠标移动
        ##鼠标移动事件，point是 GraphicsView的坐标,物理坐标
        self.__thuViewCord.setText("tv坐标：%d,%d" % (point.x(), point.y()))
        pt = self.thuview.mapToScene(point)  # 转换到Scene坐标
        self.__thuSceneCord.setText("ts坐标：%.0f,%.0f" % (pt.x(), pt.y()))

    def thu_mouseRelease(self, point):  ##缩略图点击反应
        pt = self.thuview.mapToScene(point)  # 转换到Scene坐标
        item = self.thuscene.itemAt(pt, self.thuview.transform())  # 获取光标下的图形项
        if (item == None):
            return
        pm = item.mapFromScene(pt)  # 转换为绘图项的局部坐标
        slide_path = self.viewer.slide_viewer.slide_helper.slide_path
        x = pm.x() * 4 * self.thu_level_dimensions[0] / self.thuPixmap.width()
        # y = pm.y() * 4 * self.thu_level_dimensions[1] / self.thuPixmap.height()
        y = pm.y() * 4 * self.thu_level_dimensions[0] / self.thuPixmap.width()
        slide_level = self.viewer.slide_viewer.get_thu_mouse_release_level()
        locate = self.viewer.slide_viewer.slide_helper.get_level_size(slide_level)
        locate_x = locate[0] / self.thu_level_dimensions[0] * x
        locate_y = locate[0] / self.thu_level_dimensions[0] * y
        qrectf = (
            locate_x - self.centralWidget.width(), locate_y - self.centralWidget.height(),
            self.centralWidget.width() * 2,
            self.centralWidget.height() * 2)
        # self.viewer.slide_viewer.thu_release_event(locate_x,locate_y)
        self.viewer.slide_viewer.load(SlideViewParams(slide_path, slide_level, qrectf))
        file_name = os.path.basename(slide_path)
        self.tabWidget.addTab(self.viewer, file_name)
        self.__thuItemCord.setText("thuItem：%.0f,%.0f" % (pm.x(), pm.y()))
        if self.mem > 1500:
            QPixmapCache.clear()
        else:
            return

    @pyqtSlot(bool)  ##放大
    def on_actZoomIn_triggered(self):
        self.viewer.slide_viewer.zoomIn_event()
        # QPixmapCache.clear()
        if self.mem > 1500:
            QPixmapCache.clear()
        else:
            return

    @pyqtSlot()  ##缩小
    def on_actZoomOut_triggered(self):
        self.viewer.slide_viewer.zoomOut_event()
        # QPixmapCache.clear()
        if self.mem > 1500:
            QPixmapCache.clear()
        else:
            return

    @pyqtSlot(bool)  ##选框  含checked判断的参数的情况必须用bool装饰器  直接连接到下下一页层级
    def on_act_rubberband_triggered(self, checked):
        if (checked == True):
            rubber_can = 1
            self.roi_save.setEnabled(True)
            self.pickgrid.setEnabled(False)
            self.click_tiles.setEnabled(False)
            self.circle_roi.setEnabled(False)

        else:
            rubber_can = 0
            self.roi_save.setEnabled(False)
            self.pickgrid.setEnabled(True)
            self.click_tiles.setEnabled(True)
            self.circle_roi.setEnabled(True)
            self.viewer.slide_viewer.slide_graphics.clear_level()
        self.viewer.slide_viewer.rubber_band_show(rubber_can)
        # QPixmapCache.clear()

    @pyqtSlot(bool)
    def on_circle_roi_triggered(self, checked):
        if (checked == True):
            circle_can = 1
            self.roi_save.setEnabled(False)
            self.pickgrid.setEnabled(False)
            self.click_tiles.setEnabled(False)
            self.act_rubberband.setEnabled(False)
        else:
            circle_can = 0
            self.roi_save.setEnabled(False)
            self.pickgrid.setEnabled(True)
            self.click_tiles.setEnabled(True)
            self.act_rubberband.setEnabled(True)
            self.viewer.slide_viewer.slide_graphics.clear_level()
        self.viewer.slide_viewer.circle_show(circle_can)

    @pyqtSlot(bool)
    def on_roi_save_triggered(self):

        roi_path = 'area_roi_'
        self.viewer.slide_viewer.ROI_data_save(roi_path)

    @pyqtSlot(bool)
    def on_roi_tile_save_triggered(self):

        roi_path = 'tile_roi_'
        self.viewer.slide_viewer.Tiles_cell_seg(roi_path)

    @pyqtSlot(bool)
    def on_Table_Show_triggered(self):
        print('table')
        self.viewer.slide_viewer.cell_result_show()


    @pyqtSlot(bool)
    def on_BackFilterShow_triggered(self):
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(False)
        # if (checked == True):
        #     self.viewer.get_bin_param_trans()
        #     state = True
        # else:
        #     state = False
        # self.viewer.slide_viewer.slide_graphics.update_grid_visibility(state)
        self.viewer.get_bin_param_trans()
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(True)
        print('Filter Done')

    @pyqtSlot(bool)  ##选背景  含checked判断的参数的情况必须用bool装饰器  直接连接到下下一页层级
    def on_background_triggered(self):
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(False)
        # if (checked == True):
        #     self.viewer.get_net_param_trans()
        #     state = True
        # else:
        #     state = False
        self.viewer.get_net_param_trans()
        # return False  ##返回false会导致state无法传参双点撤销失灵
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(True)
        print('background_done')
        # QPixmapCache.clear()

    @pyqtSlot(bool)  ##框选背景
    def on_pickgrid_triggered(self, checked):
        if (checked == True):
            pick_can = 1
            self.act_rubberband.setEnabled(False)
            self.click_tiles.setEnabled(False)
            self.circle_roi.setEnabled(False)
        else:
            pick_can = 0
            self.act_rubberband.setEnabled(True)
            self.click_tiles.setEnabled(True)
            self.circle_roi.setEnabled(True)
            self.viewer.slide_viewer.slide_graphics.clear_level()
        self.viewer.slide_viewer.pickgrid_show(pick_can)
        # QPixmapCache.clear()

    @pyqtSlot(bool)  ##点选背景
    def on_click_tiles_triggered(self, checked):
        if (checked == True):
            click_can = 1

            self.roi_save.setEnabled(True)
            self.pickgrid.setEnabled(False)
            self.act_rubberband.setEnabled(False)
            self.circle_roi.setEnabled(False)
        else:
            click_can = 0
            self.pickgrid.setEnabled(True)
            self.act_rubberband.setEnabled(True)
            self.circle_roi.setEnabled(True)
            self.roi_save.setEnabled(False)
            self.viewer.slide_viewer.slide_graphics.clear_level()
        self.viewer.slide_viewer.clicktiles_show(click_can)

    @pyqtSlot(bool)  ##网格添加
    def on_set_grid_triggered(self):
        # if (checked == True):
        #     state = True
        #     self.viewer.set_grid_size()
        #     #state 状态控制
        #     # self.update_CancerRate_label()
        # else:
        #     state = False
        self.time_use_bar.setValue(0)
        self.barstep = 0
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(False)
        self.pro_accept = self.viewer.set_grid_size()
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(True)

        self.level_size = self.viewer.slide_viewer.slide_helper.get_level_size(0)
        if self.exist_file == True:
            self.process_time = int (self.level_size[0] / 290 / 2.5)
        else:
            self.process_time = int (self.level_size[0] / 290)
        # self.process_time = 100

        # print(self.level_size[0])
        if self.timer.isActive():
            self.timer.stop()
        else:
            if self.pro_accept == True:
                self.timer.start(self.process_time * 10, self)
        # self.roi_tile_save.setEnabled(True)
        # self.actchoosecancertype.setEnabled(True)
        # self.actchoosecancertype.setCheckable(False)
        # self.background.setEnabled(True)
        # self.background.setCheckable(False)
        # self.BackFilterShow.setEnabled(True)
        # self.BackFilterShow.setCheckable(False)

    def timerEvent(self, a0: 'QTimerEvent') -> None:
        if self.barstep >= 100 :
            self.barstep = 0
            self.timer.stop()
            self.roi_tile_save.setEnabled(True)
            self.actchoosecancertype.setEnabled(True)
            self.actchoosecancertype.setCheckable(False)
            self.background.setEnabled(True)
            self.background.setCheckable(False)
            self.BackFilterShow.setEnabled(True)
            self.BackFilterShow.setCheckable(False)
            self.ResultShow.setEnabled(True)
            self.TilesShow.setEnabled(True)
            self.Table_Show.setEnabled(True)
            return
        self.barstep = self.barstep + 1
        # print(self.barstep)
        self.time_use_bar.setValue(self.barstep)

    @pyqtSlot(bool)  ##癌类筛选
    def on_actchoosecancertype_triggered(self):

        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(False)
        self.viewer.set_cancer_type_choose()
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(True)
        # if (checked == True):
        #     state = True
        #     self.viewer.set_cancer_type_choose()
        #
        # else:
        #     state = False
        #
        # self.viewer.slide_viewer.slide_graphics.update_grid_visibility(state)
        # slide.multiprocess_training_slides_to_images()
        # filter.multiprocess_apply_filters_to_images()
        # self.viewer.slide_viewer.slide_graphics.update_grid_visibility(state)

    @pyqtSlot(bool)
    def on_show_grid_triggered(self, state: bool):
        # if (checked == True):
        self.viewer.slide_viewer.slide_graphics.update_grid_visibility(state)
        # else:
        #     return
        ##网格显示checked判断  均为自动连接槽函数  手写槽函数不用加装饰器
        # 可以设置好checked直接用state：bool 不用单独声明checked判断了因为后面需要state的状态参量

    @pyqtSlot()  ##坐标定位
    def on_act_Location_triggered(self):
        self.viewer.go_to()

    @pyqtSlot()  ##截图
    def on_act_screenshot_triggered(self):
        print('shoot')
        self.viewer.take_screenshot()

    @pyqtSlot()  ##复原
    def on_act_reset_triggered(self):
        slide_path = self.viewer.slide_viewer.slide_helper.slide_path
        file_name = os.path.basename(slide_path)
        self.scrollArea.raise_()
        self.viewer.slide_viewer.load(SlideViewParams(slide_path))  #刷新复原
        self.tabWidget.addTab(self.viewer, file_name)

    @pyqtSlot()  ##打印 后面加上和数据库的交互
    def on_itenParams_triggered(self):
        self.viewer.print_items()

    @pyqtSlot()  ##打印 后面加上和数据库的交互
    def on_viewParams_triggered(self):
        self.viewer.print_slide_view_params()

    @property  ##支持格式
    def available_formats(self):
        whole_slide_formats = [
            "svs",
            "vms",
            "vmu",
            "ndpi",
            "scn",
            "mrx",
            "tiff",
            "svslide",
            "tif",
            "bif",
            "mrxs",
            "bif",
        ]
        pillow_formats = [
            "bmp",
            "bufr",
            "cur",
            "dcx",
            "fits",
            "fl",
            "fpx",
            "gbr",
            "gd",
            "gif",
            "grib",
            "hdf5",
            "ico",
            "im",
            "imt",
            "iptc",
            "jpeg",
            "jpg",
            "jpe",
            "mcidas",
            "mic",
            "mpeg",
            "msp",
            "pcd",
            "pcx",
            "pixar",
            "png",
            "ppm",
            "psd",
            "sgi",
            "spider",
            "tga",
            "tiff",
            "wal",
            "wmf",
            "xbm",
            "xpm",
            "xv",
        ]
        self.whole_slide_format = whole_slide_formats
        self.pillow_formats = pillow_formats
        available_formats = [*whole_slide_formats, *pillow_formats]
        available_extensions = [
            "." + available_format for available_format in available_formats
        ]
        return available_extensions

    @pyqtSlot(bool)  ##通过使用装饰器和bool实现按键的两个状态
    def on_actThu_triggered(self, checked):
        sizex = int(self.centralWidget.width() / 8)
        sizey = int(self.centralWidget.width() / 8 - 20)
        self.thuArea.setMaximumSize(
            QtCore.QSize(int(self.centralWidget.width() / 8), int(self.centralWidget.width() / 8 - 20)))
        x = self.tabWidget.sizeHint().width() - self.viewer.slide_viewer.sizeHint().width()
        y = self.tabWidget.sizeHint().height() - self.viewer.slide_viewer.sizeHint().height()
        self.thuArea.setGeometry(QtCore.QRect(x + 20, y + 30, sizex - 2, sizey - 2))
        self.thuArea.raise_()
        self.thuArea.setVisible(checked)

    @pyqtSlot(bool)  ##停靠区可见性变化
    def on_thuArea_visibilityChanged(self, visible):
        self.actThu.setChecked(visible)
        self.scrollArea.raise_()

    @pyqtSlot()
    def on_ResultShow_triggered(self):
        print('result')

        dialog = QDialog()
        dialog.setWindowTitle("前景背景分割结果汇总")
        main_layout = QVBoxLayout()
        browser1 = QWebEngineView()
        # browser2 = QWebEngineView()
        back_folder = '../dataprocess/data-index/'
        slide_path = self.viewer.slide_viewer.slide_helper.slide_path
        svs_name = slide_path.split('/')[-1][:-4]
        # back_path = os.path.join(back_folder, svs_name, "filters.html")
        path1 = os.path.realpath('../data/filters.html')
        # path2 = os.path.realpath('../data/tiles.html')
        filter_url = path1.replace('\\', '/')
        # tile_url = path2.replace('\\' , '/')
        browser1.load(QUrl(filter_url))
        # browser2.load(QUrl(tile_url))
        main_layout.addWidget(browser1)
        # main_layout.addWidget(browser2)
        ##全部添加到mainlayout
        dialog.setLayout(main_layout)
        dialog.exec() #弹窗指令

    @pyqtSlot()
    def on_TilesShow_triggered(self):

        print(55)
        self.viewer.seg_result_show()

    ##窗体自适应
    def resizeEvent(self, event):  ##窗口自适应resize
        # print('窗体自适应初始化',self.centralWidget.height(),self.centralWidget.width())
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, self.centralWidget.width(), self.centralWidget.height()))
        # self.viewer.slide_viewer.scene.setGeometry(QtCore.QRect(0, 0, self.centralWidget.width(), self.centralWidget.height()))

    @pyqtSlot(bool)  ##设置停靠区浮动性
    def on_actDockFloat_triggered(self, checked):
        self.dockWidgetOverview.setFloating(checked)

    @pyqtSlot(bool)  ##停靠区浮动性改变
    def on_dockWidget_topLevelChanged(self, topLevel):
        self.actDockFloat.setChecked(topLevel)

    @pyqtSlot(bool)  ##设置停靠区可见性
    def on_actDockVisible_triggered(self, checked):
        self.dockWidgetOverview.setVisible(checked)

    @pyqtSlot(bool)  ##停靠区可见性变化
    def on_dockWidget_visibilityChanged(self, visible):
        self.actDockVisible.setChecked(visible)

    @pyqtSlot()  ##删除所有选中的绘图项
    def on_actEdit_Delete_triggered(self):
        self.scrollArea.raise_()
        self.tabWidget.clear()

        self.barstep = 0
        self.timer.stop()
        self.time_use_bar.setValue(0)
        self.BackFilterShow.setEnabled(False)
        self.TilesShow.setEnabled(False)
        self.Table_Show.setEnabled(False)
        self.act_database.setEnabled(False)
        self.circle_roi.setEnabled(False)
        self.actThu.setEnabled(False)
        self.viewParams.setEnabled(True)
        self.actZoomIn.setEnabled(False)
        self.actZoomOut.setEnabled(False)  # 功能初始化
        self.act_rubberband.setEnabled(False)
        # self.act_rubberband.setCheckable(False)
        self.act_Location.setEnabled(False)
        self.act_screenshot.setEnabled(False)
        self.act_reset.setEnabled(False)
        self.itenParams.setEnabled(False)
        self.click_tiles.setEnabled(False)
        self.set_grid.setEnabled(False)
        self.show_grid.setEnabled(False)
        self.background.setEnabled(False)
        self.pickgrid.setEnabled(False)
        self.ResultShow.setEnabled(False)
        self.actchoosecancertype.setEnabled(False)
        self.roi_save.setEnabled(False)
        self.roi_tile_save.setEnabled(False)

    ##  ==============数据库部分自定义功能函数============##
    def __getFieldNames(self):  ##获取所有字段名称
        emptyRec = self.tabModel.record()  # 获取空记录，只有字段名
        self.fldNum = {}  # 字段名与序号的字典
        for i in range(emptyRec.count()):
            fieldName = emptyRec.fieldName(i)
            self.comboFields.addItem(fieldName)
            self.fldNum.setdefault(fieldName)
            self.fldNum[fieldName] = i
        print(self.fldNum)

    def __openTable(self):  ##打开数据表打开的db只是一个格式空的  内容需要自己写
        #数据库加载及初始化
        self.tabModel = QSqlTableModel(self, self.DB)  # 数据模型
        self.tabModel.setTable("辅助诊断结果")  # 设置数据表  在数据库中的单个数据列表名称  设置需要连接的数据表
        self.tabModel.setEditStrategy(QSqlTableModel.OnManualSubmit)  # 数据保存方式，OnManualSubmit , OnRowChange
        self.tabModel.setSort(self.tabModel.fieldIndex("患者编号"), Qt.AscendingOrder)  # 排序
        if (self.tabModel.select() == False):  # 查询数据失败
            QMessageBox.critical(self, "错误信息",
                                 "打开数据表错误,错误信息\n" + self.tabModel.lastError().text())
            return

        self.__getFieldNames()  # 获取字段名和序号

        ##字段显示名
        self.tabModel.setHeaderData(self.fldNum["患者编号"], Qt.Horizontal, "患者编号")
        self.tabModel.setHeaderData(self.fldNum["姓名"], Qt.Horizontal, "姓名")
        self.tabModel.setHeaderData(self.fldNum["性别"], Qt.Horizontal, "性别")
        self.tabModel.setHeaderData(self.fldNum["出生年月"], Qt.Horizontal, "出生年月")
        self.tabModel.setHeaderData(self.fldNum["患癌类型"], Qt.Horizontal, "患癌类型")
        self.tabModel.setHeaderData(self.fldNum["患癌子类"], Qt.Horizontal, "患癌子类")
        self.tabModel.setHeaderData(self.fldNum["概率预测"], Qt.Horizontal, "概率预测")

        self.tabModel.setHeaderData(self.fldNum["初诊建议"], Qt.Horizontal, "初诊建议")  # 这两个字段不在tableView中显示 任意长度的普通文本
        self.tabModel.setHeaderData(self.fldNum["切片"], Qt.Horizontal, "切片")

        ##创建界面组件与数据模型的字段之间的数据映射
        self.mapper = QDataWidgetMapper()
        self.mapper.setModel(self.tabModel)  # 设置数据模型
        self.mapper.setSubmitPolicy(QDataWidgetMapper.AutoSubmit)

        ##界面组件与tabModel的具体字段之间的联系     两个框架数据之间的连接方式
        self.mapper.addMapping(self.dbSpinEmpNo, self.fldNum["患者编号"])
        self.mapper.addMapping(self.dbEditName, self.fldNum["姓名"])
        self.mapper.addMapping(self.dbComboSex, self.fldNum["性别"])
        self.mapper.addMapping(self.dbEditBirth, self.fldNum["出生年月"])
        self.mapper.addMapping(self.dbComboProvince, self.fldNum["患癌类型"])
        self.mapper.addMapping(self.dbComboDep, self.fldNum["患癌子类"])
        self.mapper.addMapping(self.dbSpinSalary, self.fldNum["概率预测"])
        self.mapper.addMapping(self.dbEditMemo, self.fldNum["初诊建议"])
        self.mapper.toFirst()  # 移动到首记录

        self.selModel = QItemSelectionModel(self.tabModel)  # 选择模型
        self.selModel.currentChanged.connect(self.do_currentChanged)  # 当前项变化时触发
        self.selModel.currentRowChanged.connect(self.do_currentRowChanged)  # 选择行变化时

        self.tableView.setModel(self.tabModel)  # 设置数据模型
        self.tableView.setSelectionModel(self.selModel)  # 设置选择模型

        self.tableView.setColumnHidden(self.fldNum["初诊建议"], True)  # 隐藏列
        self.tableView.setColumnHidden(self.fldNum["切片"], True)  # 隐藏列

        ##tableView上为“性别”和“部门”两个字段设置自定义代理组件
        strList = ("男", "女")
        self.__delegateSex = QmyComboBoxDelegate()
        self.__delegateSex.setItems(strList, False)
        self.tableView.setItemDelegateForColumn(self.fldNum["性别"], self.__delegateSex)  # Combbox选择型

        strList = ("a部", "b部", "c部", "d部")
        self.__delegateDepart = QmyComboBoxDelegate()
        self.__delegateDepart.setItems(strList, True)
        self.tableView.setItemDelegateForColumn(self.fldNum["患癌子类"], self.__delegateDepart)

        ##更新actions和界面组件的使能状态
        self.act_database.setEnabled(False)

        self.actRecAppend.setEnabled(True)
        self.actRecInsert.setEnabled(True)
        self.actRecDelete.setEnabled(True)
        self.actScan.setEnabled(True)

        self.groupBoxSort.setEnabled(True)
        self.groupBoxFilter.setEnabled(True)

    @pyqtSlot()  ##选择数据库，打开数据表
    def on_act_database_triggered(self):
        dbFilename, flt = QFileDialog.getOpenFileName(self, "选择数据库文件", "",
                                                      "SQL Lite数据库(*.db *.db3)")
        if (dbFilename == ''):
            return
        self.act_database
        # 打开数据库
        self.DB = QSqlDatabase.addDatabase("QSQLITE")  # 添加 SQLITE数据库驱动
        self.DB.setDatabaseName(dbFilename)  # 设置数据库名称
        # self.DB.setHostName('55')
        # self.DB.setUserName('wch')
        # self.DB.setPassword('123')
        if self.DB.open():  # 打开数据库
            self.__openTable()  # 打开数据表
        else:
            QMessageBox.warning(self, "错误", "打开数据库失败")

    @pyqtSlot()  ##保存修改
    def on_actSubmit_triggered(self):
        res = self.tabModel.submitAll()
        if (res == False):
            QMessageBox.information(self, "消息",
                                    "数据保存错误,错误信息\n" + self.tabModel.lastError().text())
        else:
            self.actSubmit.setEnabled(False)
            self.actRevert.setEnabled(False)

    @pyqtSlot()  ##取消修改
    def on_actRevert_triggered(self):
        self.tabModel.revertAll()
        self.actSubmit.setEnabled(False)
        self.actRevert.setEnabled(False)

    @pyqtSlot()  ##添加记录
    def on_actRecAppend_triggered(self):
        self.tabModel.insertRow(self.tabModel.rowCount(), QModelIndex())  # 在末尾添加一个记录

        curIndex = self.tabModel.index(self.tabModel.rowCount() - 1, 1)  # 创建最后一行的ModelIndex
        self.selModel.clearSelection()  # 清空选择项
        self.selModel.setCurrentIndex(curIndex, QItemSelectionModel.Select)  # 设置刚插入的行为当前选择行

        currow = curIndex.row()  # 获得当前行
        self.tabModel.setData(self.tabModel.index(currow, self.fldNum["患者编号"]),
                              2000 + self.tabModel.rowCount())  # 自动生成编号
        self.tabModel.setData(self.tabModel.index(currow, self.fldNum["性别"]), "男")  ##填入默认值 每个字占四个字节

    @pyqtSlot()  ##插入记录
    def on_actRecInsert_triggered(self):
        curIndex = self.ui.tableView.currentIndex()  # QModelIndex
        self.tabModel.insertRow(curIndex.row(), QModelIndex())
        self.selModel.clearSelection()  # 清除已有选择
        self.selModel.setCurrentIndex(curIndex, QItemSelectionModel.Select)

    @pyqtSlot()  ##删除记录
    def on_actRecDelete_triggered(self):
        curIndex = self.selModel.currentIndex()  # 获取当前选择单元格的模型索引
        self.tabModel.removeRow(curIndex.row())  # 删除当前行

    @pyqtSlot()  ##清除照片
    def on_actPhotoClear_triggered(self):
        curRecNo = self.selModel.currentIndex().row()
        curRec = self.tabModel.record(curRecNo)  # 获取当前记录,QSqlRecord
        curRec.setNull("切片")  # 设置为空值
        self.tabModel.setRecord(curRecNo, curRec)
        self.dbLabPhoto.clear()  # 清除界面上的图片显示

    @pyqtSlot()  ##设置照片
    def on_actPhoto_triggered(self):
        fileName, filt = QFileDialog.getOpenFileName(self, "选择图片文件", "", "照片(*.jpg)")
        if (fileName == ''):
            return

        file = QFile(fileName)  # fileName为图片文件名
        file.open(QIODevice.ReadOnly)  # 设置为只读模式
        try:
            data = file.readAll()  # QByteArray字节列表
        finally:
            file.close()

        curRecNo = self.selModel.currentIndex().row()
        curRec = self.tabModel.record(curRecNo)  # 获取当前记录QSqlRecord
        curRec.setValue("切片", data)  # 设置字段数据
        self.tabModel.setRecord(curRecNo, curRec)

        pic = QPixmap()
        pic.loadFromData(data)
        # W = self.dbLabPhoto.width()
        W = 130
        self.dbLabPhoto.setPixmap(pic.scaledToWidth(W))  # 在界面上显示也需要QPixmap

    @pyqtSlot()  ##
    def on_actScan_triggered(self):
        if (self.tabModel.rowCount() == 0):
            return

        for i in range(self.tabModel.rowCount()):
            aRec = self.tabModel.record(i)  # 获取当前记录
            ##         salary=aRec.value("Salary").toFloat()      #错误，无需再使用toFloat()函数
            salary = aRec.value("概率预测")
            salary = salary * 1.1
            aRec.setValue("概率预测", salary)
            self.tabModel.setRecord(i, aRec)

        if (self.tabModel.submitAll()):
            QMessageBox.information(self, "消息", "计算完毕")

    @pyqtSlot(int)
    def on_comboFields_currentIndexChanged(self, index):
        if self.radioBtnAscend.isChecked():
            self.tabModel.setSort(index, Qt.AscendingOrder)
        else:
            self.tabModel.setSort(index, Qt.DescendingOrder)
        self.tabModel.select()

    @pyqtSlot()
    def on_radioBtnAscend_clicked(self):
        self.tabModel.setSort(self.comboFields.currentIndex(), Qt.AscendingOrder)
        self.tabModel.select()

    @pyqtSlot()
    def on_radioBtnDescend_clicked(self):
        self.tabModel.setSort(self.comboFields.currentIndex(), Qt.DescendingOrder)
        self.tabModel.select()

    @pyqtSlot()  ##过滤，男
    def on_radioBtnMan_clicked(self):
        self.tabModel.setFilter("性别='男'")

    @pyqtSlot()  ##数据过滤，女
    def on_radioBtnWoman_clicked(self):
        self.tabModel.setFilter("性别='女' ")

    @pyqtSlot()  ##取消数据过滤
    def on_radioBtnBoth_clicked(self):
        self.tabModel.setFilter("")

    def do_currentChanged(self, current, previous):  ##更新actPost和actCancel 的状态
        self.actSubmit.setEnabled(self.tabModel.isDirty())  # 有未保存修改时可用
        self.actRevert.setEnabled(self.tabModel.isDirty())

    def do_currentRowChanged(self, current, previous):  # 行切换时的状态控制
        self.actRecDelete.setEnabled(current.isValid())
        self.actPhoto.setEnabled(current.isValid())
        self.actPhotoClear.setEnabled(current.isValid())

        if (current.isValid() == False):
            self.dbLabPhoto.clear()  # 清除图片显示
            return

        self.mapper.setCurrentIndex(current.row())  # 更新数据映射的行号
        curRec = self.tabModel.record(current.row())  # 获取当前记录,QSqlRecord类型

        if (curRec.isNull("切片")):  # 图片字段内容为空
            self.dbLabPhoto.clear()
        else:
            ##         data=bytearray(curRec.value("Photo"))   #可以工作
            data = curRec.value("切片")  # 也可以工作
            pic = QPixmap()
            pic.loadFromData(data)
            # W = self.dbLabPhoto.size().width()
            W = 130
            self.dbLabPhoto.setPixmap(pic.scaledToWidth(W))



















##  ==============数据库部分功能函数============##


'明天做thu区域在不同屏幕分辨率下的大小配置'
' icon3.addPixmap(QtGui.QPixmap(":/images/icons/824.bmp")' \
'.scaled(16, 16, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation),' \
' QtGui.QIcon.Normal, QtGui.QIcon.Off)'
'图标icon的大小修改'
