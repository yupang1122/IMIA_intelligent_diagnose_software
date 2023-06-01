from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import PyQt5
import os
import time
import openslide
import easygui
from typing import List, Tuple
import cv2
import numpy as np
from PIL.ImageQt import ImageQt
from PyQt5 import QtCore
from slide_viewer.SlideGraphicsScene import SlideGraphicsScene
from slide_viewer.SlideGraphicsView import SlideGraphicsView
from slide_viewer.common.SlideHelper import SlideHelper
from slide_viewer.common.SlideViewParams import SlideViewParams
from slide_viewer.common.utils import point_to_str
from slide_viewer.common.level_builders import build_rects_and_color_alphas_for_background, \
    build_rects_and_color_alphas_for_type_select, \
    build_rects_and_color_alphas_for_grid, build_rects_and_color_alphas_for_filter
from slide_viewer.common.screenshot_builders import build_screenshot_image
from slide_viewer.graphics.SlideGraphicsItemGroup import SlideGraphicsItemGroup
from cell_detection.nuclei_DS import unet_process, unet_tile_process
from chart_gui.myMainWindow import QmyMainWindow
from PyQt5.QtWidgets import QGraphicsView


class SlideViewer(QWidget):
    # eventSignal = pyqtSignal(QEvent)
    # leftMouseButtonPressed = pyqtSignal(float, float)
    # rightMouseButtonPressed = pyqtSignal(float, float)
    # leftMouseButtonReleased = pyqtSignal(float, float)
    # rightMouseButtonReleased = pyqtSignal(float, float)
    # leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    # rightMouseButtonDoubleClicked = pyqtSignal(float, float)

    def __init__(self, parent: QWidget = None, viewer_top_else_left=True):
        super().__init__(parent)

        self.grid_ratio = 2
        # self.wheeltime = 0  ##滚轮次数
        self.Zoom_pos = PyQt5.QtCore.QPoint(0, 0)  ##按键放大的参数指标
        self.init_view()
        self.init_labels(word_wrap=viewer_top_else_left)
        self.init_layout(viewer_top_else_left)
        self.rubber_can = 0  ##初始化
        self.pick_can = 0
        self.click_can = 0
        self.circle_can = 0
        self.distance = (0, 0)
        self.start_point_x = 0
        self.start_point_y = 0
        self.rect_cell_result = []
        self.circle_flag = False  # 通过定义bool控制标圈时的移动记值状态
        # print('distance', self.distance)

    def init_view(self):  ##view 初始化
        self.scene = SlideGraphicsScene()
        self.view = SlideGraphicsView(self.scene)
        self.view.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.view.viewport().installEventFilter(self)
        self.view.setMouseTracking(True)  ##位置追踪
        self.view.setDragMode(QGraphicsView.RubberBandDrag)  ##拖拽操作
        # self.view.setContextMenuPolicy(Qt.CustomContextMenu)  ##右键开放策略 主页已经进行了设置
        # 这里不再设置保证在上层可以进行页面设置
        # self.view.customContextMenuRequested.connect(self.right_menu_show)
        # QRubberBand.Line=100  #线
        # QRubberBand.Rectangle=1 #矩形
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        ##rubberband 功能仅是选框实际是selectedrectgraphicsitem函数的qpen实现的功能
        self.mouse_press_view = QPoint()
        self.mouse_move_view = QPoint()
        self.view.horizontalScrollBar().sliderMoved.connect(self.on_view_changed)
        self.view.verticalScrollBar().sliderMoved.connect(self.on_view_changed)
        self.scale_initializer_deffered_function = None
        self.slide_view_params = None
        self.slide_helper = None

    def init_labels(self, word_wrap):  ##状态栏参数设置
        # word_wrap = True
        self.level_downsample_label = QLabel()
        self.level_downsample_label.setWordWrap(word_wrap)
        self.level_downsample_label.setMinimumWidth(60)
        self.level_size_label = QLabel()
        self.level_size_label.setWordWrap(word_wrap)
        self.level_size_label.setMinimumWidth(60)
        self.selected_rect_label = QLabel()
        self.selected_rect_label.setWordWrap(word_wrap)
        self.selected_rect_label.setMinimumWidth(60)
        self.mouse_pos_scene_label = QLabel()
        self.mouse_pos_scene_label.setWordWrap(word_wrap)
        self.mouse_pos_scene_label.setMinimumWidth(60)
        self.view_rect_scene_label = QLabel()
        self.view_rect_scene_label.setWordWrap(word_wrap)
        self.view_rect_scene_label.setMinimumWidth(60)

        self.labels_layout = QHBoxLayout()
        self.labels_layout.setAlignment(Qt.AlignTop)

        self.labels_layout.addWidget(self.level_downsample_label)
        self.labels_layout.addWidget(self.level_size_label)
        self.labels_layout.addWidget(self.mouse_pos_scene_label)
        self.labels_layout.addWidget(self.selected_rect_label)
        self.labels_layout.addWidget(self.view_rect_scene_label)

        self.cancer_type_label = QLabel()
        self.cancer_type_label.setWordWrap(word_wrap)
        self.cancer_type_label.setMinimumWidth(60)
        self.labels_layout.addWidget(self.cancer_type_label)
        self.cancer_rate_label = QLabel()
        self.cancer_rate_label.setWordWrap(word_wrap)
        self.cancer_rate_label.setMinimumWidth(60)
        self.labels_layout.addWidget(self.cancer_rate_label)

    # 布局初始化
    def init_layout(self, viewer_top_else_left=True):
        main_layout = QVBoxLayout(self) if viewer_top_else_left else QHBoxLayout(self)
        main_layout.addWidget(self.view)
        main_layout.addLayout(self.labels_layout)
        # main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

    def load(
            self,
            slide_view_params: SlideViewParams,
            preffered_rects_count=2000,
            zoom_step=1.15,
    ):
        self.distance = (0, 0)
        # print('distance', self.distance[1])
        self.zoom_step = zoom_step
        self.slide_view_params = slide_view_params
        self.slide_helper = SlideHelper(slide_view_params.slide_path)  ##先把位置信息给了
        self.slide_graphics = SlideGraphicsItemGroup(
            slide_view_params, self.distance, preffered_rects_count
        )  ##添加区块图元
        self.scene.clear()
        self.scene.addItem(self.slide_graphics)
        self.scene.clearSelection()
        self.slide_graphics.setSelected(True)
        self.slide_graphics.setFlag(QGraphicsItem.ItemIsFocusable)  ##针对窗体item进行设置
        # self.slide_graphics.setFlag(QGraphicsItem.ItemIsMovable)
        self.slide_graphics.setFlag(QGraphicsItem.ItemIsSelectable)  ##可移动性设置
        if self.slide_view_params.level == -1 or self.slide_view_params.level is None:
            self.slide_view_params.level = self.slide_helper.max_level
        self.slide_graphics.update_visible_level(self.slide_view_params.level)
        self.scene.setSceneRect(
            self.slide_helper.get_rect_for_level(self.slide_view_params.level)
        )

        def scale_initializer_deffered_function():
            self.view.resetTransform()
            if self.slide_view_params.level_rect:

                self.view.fitInView(
                    QRectF(*self.slide_view_params.level_rect), Qt.KeepAspectRatio
                )
            else:
                start_margins = QMarginsF(200, 200, 200, 200)
                start_image_rect_ = self.slide_helper.get_rect_for_level(
                    self.slide_view_params.level
                )
                self.view.fitInView(
                    start_image_rect_ + start_margins, Qt.KeepAspectRatio
                )

        self.scale_initializer_deffered_function = scale_initializer_deffered_function
        ##实时位置比例获取
        ##算出448像素下对应采样层级的比例进行序号对应

    def Tiles_cell_seg(self, roi_path):
        dialog = QDialog()
        dialog.setWindowTitle("细胞分割方式")
        main_layout = QVBoxLayout()
        box_a = QComboBox()
        box_a.addItem('滤波器轮廓检测')
        box_a.addItem('深度网络U-NET')
        box_layout = QHBoxLayout()
        box_layout.addWidget(box_a)
        box_final_layout = QFormLayout()
        box_final_layout.addRow("分割方式:", box_layout)
        main_layout.addLayout(box_final_layout)

        magnification = QSpinBox()
        magnification.setMaximum(100)
        magnification.setMinimum(1)
        magnification.setValue(30)

        ratio_a = QSpinBox()
        ratio_a.setMaximum(500)
        ratio_a.setMinimum(0)
        ratio_a.setValue(200)

        ratio_b = QSpinBox()
        ratio_b.setMaximum(3000)
        ratio_b.setMinimum(1000)
        ratio_b.setValue(2000)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(magnification)
        vertical_layout.addWidget(ratio_a)
        vertical_layout.addWidget(ratio_b)
        parameter_layout = QFormLayout()
        parameter_layout.addRow("过滤比例:", magnification)
        parameter_layout.addRow("面积下限:", ratio_a)
        parameter_layout.addRow("面积上限:", ratio_b)

        main_layout.addLayout(parameter_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        ##全部添加到mainlayout
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        res = dialog.exec()
        tile_size = 224
        svs_name = self.slide_helper.slide_path.split('/')[-1][:-4]
        index_path = os.path.join('../dataprocess/data-index/', svs_name, str(tile_size))  ##背景筛选信息位置 创建这个文件夹
        tum_index = index_path + '/' + 'tum_index.txt'
        cell_numbers = 0
        with open(tum_index, 'r') as f:
            annotations = f.readlines()
        for annotation in annotations:
            path = annotation.replace('\n', '')
            img = cv2.imread(path)
            if res == QDialog.Accepted:
                if box_a.currentText() == '滤波器轮廓检测':
                    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blockSize = 31
                    th4 = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                                blockSize, 5)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    iClose = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel)  # 闭运算
                    contours, hirarchy = cv2.findContours(iClose, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    # print("细胞核区域：" + format(np.size(contours)))
                    count, contours_check = self.countAll(contours, 0,0)
                    cell_numbers = int(format(count)) + cell_numbers
                    print("计数结果：" + format(count))
                else:
                    contours = unet_process(img)
                    count, contours_check = self.countAll(contours, 0, 0)
                    cell_numbers = int(format(count)) + cell_numbers
                    print("计数结果：" + format(count))
            else:
                return
        self.slide_view_params.tile_cell_num = cell_numbers
        print('细胞数：', cell_numbers, '癌类数:', len(annotations))
        easygui.msgbox(str(int(cell_numbers / 2)), '细胞核数')
        return





    def ROI_data_save(self, roi_path):
        dialog = QDialog()
        dialog.setWindowTitle("细胞分割")
        main_layout = QVBoxLayout()
        box_a = QComboBox()
        box_a.addItem('滤波器轮廓检测')
        box_a.addItem('深度网络U-NET')
        box_layout = QHBoxLayout()
        box_layout.addWidget(box_a)
        box_final_layout = QFormLayout()
        box_final_layout.addRow("分割方式:", box_layout)
        main_layout.addLayout(box_final_layout)
        magnification = QSpinBox()
        magnification.setMaximum(100)
        magnification.setMinimum(1)
        magnification.setValue(30)

        ratio_a = QSpinBox()
        ratio_a.setMaximum(500)
        ratio_a.setMinimum(0)
        ratio_a.setValue(200)

        ratio_b = QSpinBox()
        ratio_b.setMaximum(3000)
        ratio_b.setMinimum(1000)
        ratio_b.setValue(2000)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(magnification)
        vertical_layout.addWidget(ratio_a)
        vertical_layout.addWidget(ratio_b)
        parameter_layout = QFormLayout()
        parameter_layout.addRow("过滤比例:", magnification)
        parameter_layout.addRow("面积下限:", ratio_a)
        parameter_layout.addRow("面积上限:", ratio_b)

        main_layout.addLayout(parameter_layout)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        ##全部添加到mainlayout
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        res = dialog.exec()
        if self.slide_view_params.selected_rect_0_level != None and res == QDialog.Accepted:
            level_downsample = self.slide_helper.get_downsample_for_level(
                self.slide_view_params.level
            )
            a = int(self.start_point_x * level_downsample)
            b = int(self.start_point_y * level_downsample)
            print(a, b)
            with openslide.open_slide(self.slide_view_params.slide_path) as slide:
                a, b, c, d = self.slide_view_params.selected_rect_0_level
                tile_pilimage = slide.read_region(
                    (int(a), int(b)),
                    0,
                    (int(c), int(d)),
                )
                pixmap = ImageQt(tile_pilimage)
                pix = QPixmap.fromImage(pixmap)
                now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                save_path = os.path.join('../dataprocess/ROI_data/')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                roi_img = pix.save(save_path + str(roi_path) + str(now_time) + '.jpg')
                img = cv2.cvtColor(np.array(tile_pilimage), cv2.COLOR_RGB2BGR)

                if res == QDialog.Accepted:
                    if box_a.currentText() == '滤波器轮廓检测':

                        contours_0_level, cell_num, contours_check = self.opencv_cell_detection(img, a, b)
                        print('lvbook')
                    else :
                        print('unetok')
                        # contours = unet_process(img, a, b)
                        # count, contours_check = self.countAll(contours)
                        contours = unet_process(img)
                        count, contours_check = self.countAll(contours, 0,0)

                        cell_num = count / 2
                        self.slide_view_params.cell_num = format(count)

                        contours = contours_check
                        contours_0_level = []
                        for i in range(len(contours)):
                            for j in range(len(contours[i])):
                                c = []
                                pointx = int(contours[i][j][0][0] + a)
                                pointy = int(contours[i][j][0][1] + b)
                                c.append(pointx)
                                c.append(pointy)
                                # contour_point = QPoint(pointx, pointy)
                                # contours_0_level.append(contour_point)
                                contours_0_level.append(c)
                        print('points', contours_0_level)
                    print(len(contours_check))
                    area_mean = []
                    for z in range(np.size(contours_check)):
                            area = cv2.contourArea(contours_check[z])  # 计算闭合轮廓面积
                            # print(area)
                            area_mean.append(area)
                    area_param = []
                    amean = int(np.mean(area_mean))
                    # b = np.argmax(area_mean) #最大索引
                    amax = int(np.amax(area_mean))
                    amin = int(np.amin(area_mean))
                    # print(a, c, d)
                    area_param.append(amean)
                    area_param.append(amax)
                    area_param.append(amin)
                    self.slide_graphics.clear_level()
                    self.slide_view_params.contours_0_level = contours_0_level
                    self.slide_graphics.update_paint_contours_0_level(self.slide_view_params.contours_0_level)
                    self.update_labels()
                    self.scene.invalidate()

                else:
                    return
                easygui.msgbox(str(int(cell_num)), '细胞核数')

                ss = np.array(self.slide_view_params.selected_rect_0_level)

                aa = ss.astype(np.int)
                bb = aa.tolist()
                rect_cell_result = []
                rect_cell_result.append(bb)
                rect_cell_result.append(int(cell_num))
                rect_cell_result.append(area_param)
                self.slide_view_params.rect_cell_result.append(rect_cell_result)
                print(self.slide_view_params.rect_cell_result)

            return
        else:
            return

    def opencv_cell_detection(self, img, a, b):
        print('cv', a, b)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = img.copy()  # 避免描边区域和圈图圆圈同时出现
        # img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # ret, th1 = cv2.threshold(grayImage, 177, 255, cv2.THRESH_BINARY)
        blockSize = 31
        value = -1
        # th3 = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize,
        #                             value)
        th4 = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    blockSize, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        iClose = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel)  # 闭运算
        contours, hirarchy = cv2.findContours(iClose, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        print("细胞核区域：" + format(np.size(contours)))
        res = cv2.drawContours(img, contours, -1, (0, 0, 255), 2) ##实际不需要  只绘制mask在界面里即可
        cv2.imshow("seg_res", res)

        area_mean = []
        for i in range(np.size(contours)):
            area = cv2.contourArea(contours[i])  # 计算闭合轮廓面积
            area_mean.append(area)
        max_index = np.argmax(area_mean)
        max_area = np.amax(area_mean)
        print(max_index,max_area)
        count, contours_check = self.countAll(contours, max_area, max_index)
        # print(count, contours_check)

        print("计数结果：" + format(count))
        self.slide_view_params.cell_num = format(count)

        contours = contours_check
        contours_0_level = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                c = []
                pointx = int(contours[i][j][0][0] + a)
                pointy = int(contours[i][j][0][1] + b)
                c.append(pointx)
                c.append(pointy)
                # contour_point = QPoint(pointx, pointy)
                # contours_0_level.append(contour_point)
                contours_0_level.append(c)
        print('points', contours_0_level)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cell_num = count
        return contours_0_level, cell_num, contours_check

    def countAll(self, contours, max_count, max_index):
        # min_area = 100
        # max_area = 2000
        contours_check = []
        contours_delmax = []

        count = 0
        area_mean = []
        for i in range(np.size(contours)):
            if max_count != 0 and i == max_index:
                continue
            else:
                area = cv2.contourArea(contours[i])  # 计算闭合轮廓面积
                area_mean.append(area)
                contours_delmax.append(contours[i])

        a = np.mean(area_mean)
        b = np.argmax(area_mean)
        d = np.amax(area_mean)
        c = np.amin(area_mean)
        print(a,b,c,d)

        min_area = 150
        max_area = a + 8000

        for i in range(np.size(contours_delmax)):
            area = cv2.contourArea(contours_delmax[i])  # 计算闭合轮廓面积
            if (area < min_area) or (area > max_area) :
                continue
            else:
                count = count + 1
                # (x, y), radius = cv2.minEnclosingCircle(contours[i])
                # (x, y, radius) = np.int0((x, y, radius))
                contours_check.append(contours[i])


        return count, contours_check

    def cell_result_show(self):
        slide_path = self.slide_view_params.slide_path
        print(slide_path)
        svs_name = slide_path.split('/')[-1][:-4]
        print(svs_name)


        rect_cell_result = self.slide_view_params.rect_cell_result
        print(rect_cell_result)
        cellwindow = QmyMainWindow(rect_cell_result,svs_name)        #创建主窗体
        cellwindow.exec()
        # cellwindow.setAttribute(Qt.WA_DeleteOnClose)
        #
        # cellwindow.show()

    ##背景筛选
    def bin_show(self, bin_size, unet_use):
        print('bin_show', bin_size)
        rects, cols, rows, color_alphas = \
            build_rects_and_color_alphas_for_filter(
                (bin_size, bin_size),
                self.slide_helper.get_level_size(0),
                self.slide_helper.slide_path,
                bin_size)
        print(len(rects),len(color_alphas))
        self.slide_graphics.update_bin_rects_0_level(
            rects, color_alphas, bin_size,  unet_use
        )
        print('filterok', len(color_alphas))
        return

    def background_show(self, tile_size):
        grid_ratio, grid_size, mag = self.slide_helper.get_best_mag_for_slide
        print('backgroundshow', tile_size)
        rects, color_alphas, cols, rows, cancer_type, mouse_rects_dict = \
            build_rects_and_color_alphas_for_background(
                (tile_size * grid_ratio, tile_size * grid_ratio),
                self.slide_helper.get_level_size(0),
                self.slide_helper.slide_path,
                self.distance,
                tile_size,
            )

        self.slide_graphics.update_grid_rects_0_level(
            rects, color_alphas, self.distance, cancer_type, mouse_rects_dict, tile_size
        )
        print('buildok', len(color_alphas))
        return

    def cancer_type_show(self, select_type, clarity, cancer_rate, tile_size):

        grid_ratio, grid_size, mag = self.slide_helper.get_best_mag_for_slide

        rects, color_alphas, cols, rows, cancer_type, \
        rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, \
        mouse_rects_dict, tiles_num = \
            build_rects_and_color_alphas_for_type_select(
                (tile_size * grid_ratio, tile_size * grid_ratio),
                self.slide_helper.get_level_size(0),
                self.slide_helper.slide_path,
                self.distance,
                cancer_rate,
                tile_size
            )
        # type_rate = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9]
        # type = []
        # for x in select_type:
        #     type.append(type_rate[x-1])
        # print(len(type))

        self.slide_graphics.update_grid_rects_type_select(
            rects, color_alphas, self.distance, cancer_type, select_type,
            rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, clarity,
            mouse_rects_dict, tile_size, tiles_num
        )
        print('buildok', len(color_alphas), len(rate9))



    ##rects已经重新构建

    def pickgrid_show(self, pick_can):
        self.pick_can = pick_can

    def clicktiles_show(self, click_can):
        self.click_can = click_can

    def rubber_band_show(self, rubber_can):
        self.rubber_can = rubber_can

    def circle_show(self, circle_can):
        self.circle_can = circle_can

    def menuSlot(self, act):
        # print(act.text())
        if act.text() == '×2':
            self.update_scale(self.Zoom_pos, 2)
            self.on_view_changed()
            return True
        elif act.text() == '×5':
            self.update_scale(self.Zoom_pos, 5)
            self.on_view_changed()
            return True
        elif act.text() == '×10':
            self.update_scale(self.Zoom_pos, 10)
            self.on_view_changed()
            return True
        elif act.text() == 'zoom_in':
            self.zoomIn_event()
        elif act.text() == 'zoom_out':
            self.zoomOut_event()

    def eventFilter(self, qobj: QObject, event: QEvent):
        # self.eventSignal.emit(event)
        event_processed = False
        # print("size when event: ", event, event.type(), self.view.viewport().size())
        if isinstance(event, QShowEvent):
            if self.scale_initializer_deffered_function:
                # self.update_labels()
                self.scale_initializer_deffered_function()
                self.on_view_changed()
                self.scale_initializer_deffered_function = None
        elif isinstance(event, QWheelEvent):
            event_processed = self.process_viewport_wheel_event(event)
        elif isinstance(event, QMouseEvent):
            event_processed = self.process_mouse_event(event)
        return event_processed

    def zoomIn_event(self):
        zoom_in = self.zoom_step
        zoom_ = zoom_in
        self.update_scale(self.Zoom_pos, zoom_)
        self.on_view_changed()
        return True

    def zoomOut_event(self):
        zoom_out = 1 / self.zoom_step
        zoom_ = zoom_out
        self.update_scale(self.Zoom_pos, zoom_)
        self.on_view_changed()
        return True

    def get_thu_mouse_release_level(self):
        return self.slide_view_params.level

    def thu_release_event(self, locate_x, locate_y):
        zoom_ = 1
        self.update_scale(PyQt5.QtCore.QPoint(locate_x, locate_y), zoom_)
        self.on_view_changed()
        return True

    def process_viewport_wheel_event(self, event: QWheelEvent):
        # print("size when wheeling: ", self.view.viewport().size())
        zoom_in = self.zoom_step
        zoom_out = 1 / zoom_in
        if event.angleDelta().y() > 0:
            zoom_ = zoom_in
        else:
            zoom_ = zoom_out
        self.Zoom_pos = event.pos()
        self.update_scale(event.pos(), zoom_)
        event.accept()
        self.on_view_changed()

        return True

    def process_mouse_event(self, event: QMouseEvent):  ##主要是rubberband功能
        pt = self.view.mapToScene(event.pos())  # 转换到Scene坐标

        level_downsample = self.slide_helper.get_downsample_for_level(
            self.slide_view_params.level
        )
        pm = self.slide_graphics.mapFromScene(pt)
        grid_ratio, real_size, mag = self.slide_helper.get_best_mag_for_slide

        tile_size = self.slide_view_params.tile_size
        if tile_size == None:
            grid_size = real_size
        else:
            grid_size = int(tile_size * grid_ratio)
        # print(grid_size)

        self.mouse_pos_scene_label.setText(
            "mouse_scene: " + point_to_str(pt)
        )
        if self.slide_view_params.grid_visible == True:
            # cancer_type = self.slide_view_params.cancer_type
            # print(self.slide_view_params.get_rect)  #address有返回值
            col = int(pm.x() * level_downsample / grid_size)
            row = int(pm.y() * level_downsample / grid_size)  # 位置在第几个格子
            key = str(col) + '_' + str(row)
            mouse_rects_dict = self.slide_view_params.mouse_rects_dict
            if mouse_rects_dict:
                cancer_value = mouse_rects_dict.get(key)
                if cancer_value != None:
                    if cancer_value[1] == 0:
                        mouse_type = 'BACK'
                    elif cancer_value[1] == 1:
                        mouse_type = 'ADI'
                    elif cancer_value[1] == 2:
                        mouse_type = 'BACK'
                    elif cancer_value[1] == 3:
                        mouse_type = 'DEB'
                    elif cancer_value[1] == 4:
                        mouse_type = 'LYM'
                    elif cancer_value[1] == 5:
                        mouse_type = 'MUC'
                    elif cancer_value[1] == 6:
                        mouse_type = 'MUS'
                    elif cancer_value[1] == 7:
                        mouse_type = 'NORM'
                    elif cancer_value[1] == 8:
                        mouse_type = 'STR'
                    elif cancer_value[1] == 9:
                        mouse_type = 'TUM'

                    self.cancer_rate_label.setText("cancer rate: " + str(cancer_value[0]))
                    self.cancer_type_label.setText("cancer type: " + mouse_type)

        # print(pm.x() * level_downsample, pm.y() * level_downsample)  ##实时位置
        if self.slide_helper is None:
            return False  # 保证图片信息传入
        if self.rubber_can == 1:
            if event.button() == Qt.MiddleButton:
                if event.type() == QEvent.MouseButtonPress:

                    self.slide_graphics.update_grid_visibility(
                        not self.slide_graphics.slide_view_params.grid_visible
                    )
                    return True
            elif event.button() == Qt.LeftButton:
                if event.type() == QEvent.MouseButtonPress:
                    self.start_point_x = pm.x()
                    print(self.start_point_x)
                    self.start_point_y = pm.y()
                    self.mouse_press_view = QPoint(event.pos())
                    print(self.mouse_press_view)
                    self.rubber_band.setGeometry(QRect(self.mouse_press_view, QSize()))  # 设置鼠标拖动选框
                    self.rubber_band.show()
                    return True
                elif event.type() == QEvent.MouseButtonRelease:
                    self.rubber_band.hide()  # 选择框
                    self.remember_selected_rect_params()
                    self.slide_graphics.update_selected_rect_0_level(
                        self.slide_view_params.selected_rect_0_level  # 返回最底层框选区域
                    )  # 进入 selectrectgraphicsitem
                    # 这个地方运行完了调用的pen 应该使用的invalidate
                    # 松开鼠标图片保存功能
                    self.update_labels()
                    self.scene.invalidate()
                    # 调用Invalidate来重绘Item层或调用update  # 此时的重绘会发生在boundingRect()函数返回的矩形区域中
                    # 同时这个矩形区域也是响应事件的区域，在没有重载shape()的情况下还会用于碰撞检测。
                    # 此处运行完了才完成画框 绘制self.qrectf的值
                    return True
            elif event.type() == QEvent.MouseMove:
                if not self.mouse_press_view.isNull():
                    self.rubber_band.setGeometry(
                        QRect(self.mouse_press_view, event.pos()).normalized()
                    )
                return True
            return False

        elif self.circle_can == 1:
            if event.button() == Qt.LeftButton:
                if event.type() == QEvent.MouseButtonPress:
                    self.mouse_press_view = QPoint(event.pos())
                    self.start_point_x = pm.x()
                    self.start_point_y = pm.y()
                    self.x1 = pm.x()
                    self.y1 = pm.y()
                    self.circle_flag = True
                elif event.type() == QEvent.MouseButtonRelease:
                    self.circle_flag = False
                    print(self.start_point_x, self.start_point_y, self.x1, self.y1,
                          int(int((self.start_point_x - self.x1) * level_downsample) ** 2 + int((self.start_point_y - self.y1) * level_downsample) ** 2))
                    if int(int((self.start_point_x - self.x1) * level_downsample) ** 2 + int((self.start_point_y - self.y1) * level_downsample) ** 2) <= 20000:

                        draw_line_0_level = (int(self.x1 * level_downsample), int(self.y1 * level_downsample),
                                             int(self.start_point_x * level_downsample), int(self.start_point_y * level_downsample))
                        self.slide_view_params.draw_line_0_level = draw_line_0_level
                        self.slide_graphics.update_draw_line_0_level(self.slide_view_params.draw_line_0_level)
                        self.update_labels()
                        self.scene.invalidate()
                    else:
                        return True
                    return True
            elif event.type() == QEvent.MouseMove:
                if self.circle_flag:
                    self.x0 = self.x1
                    self.y0 = self.y1
                    self.x1 = pm.x()
                    self.y1 = pm.y()
                    # 完成了位置信息的刷新
                    ##随鼠标移动循环赋值
                    draw_line_0_level = (int(self.x0 * level_downsample), int(self.y0 * level_downsample),
                                         int(self.x1 * level_downsample),int(self.y1 * level_downsample))
                    self.slide_view_params.draw_line_0_level = draw_line_0_level
                    self.slide_graphics.update_draw_line_0_level(self.slide_view_params.draw_line_0_level)
                    self.update_labels()
                    self.scene.invalidate()
                return True
            return False

        elif self.pick_can == 1:
            #     return False
            # else:
            if event.button() == Qt.LeftButton:
                if event.type() == QEvent.MouseButtonPress:
                    self.mouse_press_view = QPoint(event.pos())
                    self.rubber_band.setGeometry(QRect(self.mouse_press_view, QSize()))  ##设置鼠标拖动选框
                    self.rubber_band.show()
                    return True
                elif event.type() == QEvent.MouseButtonRelease:
                    self.rubber_band.hide()
                    self.remember_selected_rect_params()
                    self.slide_graphics.update_pick_rect_0_level(
                        self.slide_view_params.selected_rect_0_level  ##返回最底层框选区域
                    )  # 进入selectrectgraphicsitem  传入数据
                    # print(self.slide_view_params.selected_rect_0_level)   ##获取的是0level范围
                    # self.update_pos_tiles_list(self.slide_view_params.selected_rect_0_level)
                    # 更新list
                    self.update_labels()
                    self.scene.invalidate()
                    return True
            elif event.type() == QEvent.MouseMove:
                if not self.mouse_press_view.isNull():
                    self.rubber_band.setGeometry(
                        QRect(self.mouse_press_view, event.pos()).normalized()
                    )
                return True
            return False
        elif self.click_can == 1:
            if event.button() == Qt.LeftButton:
                if event.type() == QEvent.MouseButtonPress:
                    return True
                elif event.type() == QEvent.MouseButtonRelease:
                    selected_qrectf_0_level = QRectF(
                        int(pm.x() * level_downsample / grid_size) * grid_size,
                        int(pm.y() * level_downsample / grid_size) * grid_size,
                        grid_size, grid_size
                    )  # 设置点选框位置  此处不对gridsize取整
                    self.slide_view_params.selected_rect_0_level = selected_qrectf_0_level.getRect()
                    self.slide_graphics.update_click_rect_0_level(self.slide_view_params.selected_rect_0_level)
                    # self.update_click_tiles_list(self.slide_view_params.selected_rect_0_level)
                    self.update_labels()  # 返回label
                    self.scene.invalidate()
                    return True
            return False
        else:
            if event.button() == Qt.LeftButton:
                # self.slide_graphics.setFlag(QGraphicsItem.ItemIsMovable)  # 移动许可
                # self.slide_graphics.setFlag(QGraphicsItem.ItemPositionChange)
                # self.slide_graphics.setFlag(QGraphicsItem.ItemPositionHasChanged)
                # 全部释放将无法运行拖动功能   后面添加一个条件判断所存

                if event.type() == QEvent.MouseButtonPress:
                    # print(self.slide_graphics.scenePos())
                    # print(self.slide_graphics.mapToParent(event.pos()))
                    a = self.view.mapToScene(event.pos())
                    # print(self.view.mapToScene(event.pos()))
                    # print(self.slide_graphics.mapFromScene(self.view.mapToScene(event.pos())))
                    # return True
                elif event.type() == QEvent.MouseButtonRelease:
                    self.distance = (self.slide_graphics.scenePos().x() * level_downsample,
                                     self.slide_graphics.scenePos().y() * level_downsample)
                    # print(self.distance)
                    b = self.view.mapToScene(event.pos())
                    # print(self.view.mapToScene(event.pos()))
                    # print(self.slide_graphics.mapFromScene(self.view.mapToScene(event.pos())))
                    # self.update_labels()  ##返回label
                    # self.scene.invalidate()
                    # self.mouse_move_distance = (self.b.x()-self.a.x(),self.b.y()-self.a.y())
                    # print(self.mouse_move_distance)
                    # return True
                    # zoom_ = 1
                    # self.update_scale(event.pos(), zoom_)
                    # print(event.pos()) #view的坐标体系
                    # self.on_view_changed()
                #     return True
                # elif event.type() == QEvent.MouseButtonRelease:
                #     self.update_labels()
                #     self.scene.invalidate()
                #     return True

            # zoom_ = 1
            # self.update_scale(event.pos(), zoom_)
            # event.accept()
            # self.on_view_changed()
            # self.update_labels()
            # self.scene.invalidate()

        return False

    def update_pos_tiles_list(self, pick_size):
        x, y, w, h = pick_size
        grid_ratio, grid_size, mag = self.slide_helper.get_best_mag_for_slide
        col = [int(x / grid_size), int((x + w) / grid_size)]
        row = [int(y / grid_size), int((y + h) / grid_size)]
        slide_name = self.slide_helper.slide_path
        svs_name = slide_name.split('/')[-1][:-4]
        index_path = os.path.join('../dataprocess/data-index/', svs_name)  ##背景筛选信息位置 创建这个文件夹
        pos_index = index_path + '/' + 'pos_index.txt'
        part_pos = open(index_path + '/' + 'part_pos.txt', 'a')
        pick_list = []
        with open(pos_index, 'r') as f:
            annotations = f.readlines()
        for pick_col in range(col[0], col[1]):
            for pick_row in range(row[0], row[1]):
                pick_list.append([pick_col, pick_row])
        for annotation in annotations:  ##循环嵌套方式
            pick_delete = annotation
            annotation = annotation.strip().split(' ')  # 去除空格根据空格进行字符化加逗号引号
            cx = int(annotation[1])
            ry = int(annotation[2])
            for a in pick_list:
                if cx == a[0]:
                    if ry == a[1]:
                        # print(len(pick_list))
                        part_pos.writelines(str(pick_delete))
                #     else:
                #         part_pos.writelines(str(pick_delete))
                # else:
                #     part_pos.writelines(str(pick_delete)) ##ok
        return

    def update_click_tiles_list(self, pick_size):
        x, y, w, h = pick_size
        grid_ratio, grid_size, mag = self.slide_helper.get_best_mag_for_slide
        slide_name = self.slide_helper.slide_path
        svs_name = slide_name.split('/')[-1][:-4]
        index_path = os.path.join('../dataprocess/data-index/', svs_name)  ##背景筛选信息位置 创建这个文件夹
        pos_index = index_path + '/' + 'pos_index.txt'
        new_pos = open(index_path + '/' + 'new_pos.txt', 'a')
        pos_list = []
        pick_col = int(x / grid_size)
        pick_row = int(y / grid_size)
        with open(pos_index, 'r') as f:
            annotations = f.readlines()
        for annotation in annotations:  # 循环嵌套方式
            click_delete = annotation
            annotation = annotation.strip().split(' ')  # 去除空格根据空格进行字符化加逗号引号
            cx = int(annotation[1])
            ry = int(annotation[2])
            if cx == pick_col:
                if ry == pick_row:
                    pos_list.append([cx, ry])
                    new_pos.writelines(str(click_delete))
            #     else:
            #         new_pos.writelines(str(click_delete))
            # else:
            #     new_pos.writelines(str(click_delete)) ##ok
        print(pick_col, pick_row)
        print(pos_list)
        return

    def remember_selected_rect_params(self):
        pos_scene = self.view.mapToScene(self.rubber_band.pos())
        rect_scene = self.view.mapToScene(self.rubber_band.rect()).boundingRect()
        downsample = self.slide_helper.get_downsample_for_level(
            self.slide_view_params.level
        )
        selected_qrectf_0_level = QRectF(
            pos_scene * downsample, rect_scene.size() * downsample
        )
        self.slide_view_params.selected_rect_0_level = selected_qrectf_0_level.getRect()

    def update_scale(self, mouse_pos: QPoint, zoom):
        old_mouse_pos_scene = self.view.mapToScene(mouse_pos)
        old_view_scene_rect = self.view.mapToScene(
            self.view.viewport().rect()
        ).boundingRect()

        old_level = self.get_best_level_for_scale(self.get_current_view_scale())
        old_level_downsample = self.slide_helper.get_downsample_for_level(old_level)
        new_level = self.get_best_level_for_scale(self.get_current_view_scale() * zoom)
        new_level_downsample = self.slide_helper.get_downsample_for_level(new_level)

        level_scale_delta = 1 / (new_level_downsample / old_level_downsample)

        r = old_view_scene_rect.topLeft()
        m = old_mouse_pos_scene
        new_view_scene_rect_top_left = (m - (m - r) / zoom) * level_scale_delta
        new_view_scene_rect = QRectF(
            new_view_scene_rect_top_left,
            old_view_scene_rect.size() * level_scale_delta / zoom,
        )

        new_scale = (
                self.get_current_view_scale()
                * zoom
                * new_level_downsample
                / old_level_downsample
        )
        transform = (
            QTransform()
                .scale(new_scale, new_scale)
                .translate(-new_view_scene_rect.x(), -new_view_scene_rect.y())
        )

        new_rect = self.slide_helper.get_rect_for_level(new_level)

        # self.scene.removeItem()
        self.scene.setSceneRect(new_rect)
        self.slide_view_params.level = new_level
        self.reset_view_transform()
        self.view.setTransform(transform, False)

        self.slide_graphics.update_visible_level(new_level)
        self.update_labels()

    def get_best_level_for_scale(self, scale):
        scene_width = self.scene.sceneRect().size().width()
        candidates = [0]
        for level in self.slide_helper.levels:
            w, h = self.slide_helper.get_level_size(level)
            if scene_width * scale <= w:
                candidates.append(level)
        best_level = max(candidates)
        return best_level

    def update_labels(self):
        level_downsample = self.slide_helper.get_downsample_for_level(
            self.slide_view_params.level
        )
        level_size = self.slide_helper.get_level_size(self.slide_view_params.level)
        self.level_downsample_label.setText(
            "level, downsample: {}, {:.0f}".format(
                self.slide_view_params.level, level_downsample
            )
        )
        self.level_size_label.setText("level_size: ({}, {})".format(*level_size))
        self.view_rect_scene_label.setText(
            "view_scene: ({:.0f},{:.0f},{:.0f},{:.0f})".format(
                *self.get_current_view_scene_rect().getRect()
            )
        )
        if self.slide_view_params.selected_rect_0_level:  ##0 level下对应的筐选范围
            self.selected_rect_label.setText(
                "selected rect (0-level): ({:.0f},{:.0f},{:.0f},{:.0f})".format(
                    *self.slide_view_params.selected_rect_0_level  ##将format指定的数以何种格式输出
                )
            )
        return self.slide_view_params.level

    def on_view_changed(self):
        if self.scale_initializer_deffered_function is None and self.slide_view_params:
            self.slide_view_params.level_rect = (
                self.get_current_view_scene_rect().getRect()
            )
        self.update_labels()

    def reset_view_transform(self):
        self.view.resetTransform()
        self.view.horizontalScrollBar().setValue(0)
        self.view.verticalScrollBar().setValue(0)

    def get_current_view_scene_rect(self):
        return self.view.mapToScene(self.view.viewport().rect()).boundingRect()

    def get_current_view_scale(self):
        scale = self.view.transform().m11()
        return scale
