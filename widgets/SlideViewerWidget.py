from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import openslide
import time
from multiprocessing import Process, Manager


from segmentation.color_resnet_tile import cut_tiles as nine_class_net
from segmentation.background_filter import background_filter_a as background_filter
from slide_viewer.common.json_utils import to_json
from slide_viewer.common.level_builders import build_rects_and_color_alphas_for_grid
from slide_viewer.common.qt.my_spin_box import MySpinBox
from slide_viewer.common.qt.myDialog import img_viewed
from chart_gui.myMainWindow import QmyMainWindow
from slide_viewer.common.screenshot_builders import build_screenshot_image
from slide_viewer.common.SlideViewParams import SlideViewParams
from slide_viewer.SlideViewer import SlideViewer
from resources.ui.SlideViewerWidget import Ui_SliderViewerWidget


# from wsi import slide
# from wsi import filter
# from wsi import tiles


class SlideViewerWidget(QWidget, Ui_SliderViewerWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        # mouse_ratio = None
        # self.img_viewer = img_viewed(555)
        self.slide_viewer = SlideViewer(viewer_top_else_left=True)
        # self.verticalLayout.addWidget(self.toolbar)
        self.verticalLayout.addWidget(self.slide_viewer)
        self.select_type = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.cancer_rate = 0
        self.clarity = 255
        self.tile_size = 224
        self.bin_size = 32
        self.unet_use = False

    @property
    def toolbar(self) -> QToolBar:
        toolbar = QToolBar()

        # set_grid_size_action = QAction(QIcon("./resources/icons/grid.png"), "Grid Size", self)
        set_grid_size_action = QAction("Grid Size", self)
        set_grid_size_action.triggered.connect(self.set_grid_size)
        toolbar.addAction(set_grid_size_action)

        show_grid_action = QAction("Show Grid", self)
        show_grid_action.setCheckable(True)
        show_grid_action.triggered.connect(self.show_grid)
        toolbar.addAction(show_grid_action)

        go_to_action = QAction("Location", self)
        go_to_action.triggered.connect(self.go_to)
        toolbar.addAction(go_to_action)

        take_screenshot_action = QAction("Screenshot", self)
        take_screenshot_action.triggered.connect(self.take_screenshot)
        toolbar.addAction(take_screenshot_action)

        print_items_action = QAction("ItemParams", self)
        print_items_action.triggered.connect(self.print_items)
        toolbar.addAction(print_items_action)

        print_slide_view_params_action = QAction("SlideViewParams", self)
        print_slide_view_params_action.triggered.connect(self.print_slide_view_params)
        toolbar.addAction(print_slide_view_params_action)

        show_rubber_band_action = QAction("rubber_band", self)
        show_rubber_band_action.setCheckable(True)
        show_rubber_band_action.triggered.connect(self.show_rubber_band)
        toolbar.addAction(show_rubber_band_action)
        return toolbar

    def set_cancer_type_choose(self):
        dialog = QDialog()
        dialog.setWindowTitle("类型选择")

        main_layout = QVBoxLayout()

        box_a = QComboBox()
        box_a.addItem('224')
        box_a.addItem('112')
        box_a.addItem('56')
        box_layout = QHBoxLayout()
        box_layout.addWidget(box_a)
        box_final_layout = QFormLayout()
        box_final_layout.addRow("采样块尺寸:", box_layout)
        main_layout.addLayout(box_final_layout)

        select_all = QCheckBox('全选')
        select_a = QCheckBox('脂肪组织')
        select_b = QCheckBox('背景区域')
        select_c = QCheckBox('组织碎片')
        select_d = QCheckBox('淋巴细胞')
        select_e = QCheckBox('粘液区域')
        select_f = QCheckBox('肌肉细胞')
        select_g = QCheckBox('正常细胞')
        select_h = QCheckBox('基质细胞')
        select_i = QCheckBox('肿瘤细胞')

        all_label = QLabel('类型选择')
        all_label.setAlignment(Qt.AlignCenter)
        all_label.setStyleSheet("background-color:hsv(0, 0, 255)")

        a_label = QLabel('类型选择')
        a_label.setAlignment(Qt.AlignCenter)
        a_label.setStyleSheet("background-color:hsv(239, 244, 255)")

        b_label = QLabel('类型选择')
        b_label.setAlignment(Qt.AlignCenter)
        b_label.setStyleSheet("background-color:hsv(303, 129, 255)")

        c_label = QLabel('类型选择')
        c_label.setAlignment(Qt.AlignCenter)
        c_label.setStyleSheet("background-color:hsv(272, 252, 255)")

        d_label = QLabel('类型选择')
        d_label.setAlignment(Qt.AlignCenter)
        d_label.setStyleSheet("background-color:hsv(217, 106, 255)")

        e_label = QLabel('类型选择')
        e_label.setAlignment(Qt.AlignCenter)
        e_label.setStyleSheet("background-color:hsv(168, 223, 255)")

        f_label = QLabel('类型选择')
        f_label.setAlignment(Qt.AlignCenter)
        f_label.setStyleSheet("background-color:hsv(120, 253, 255)")

        g_label = QLabel('类型选择')
        g_label.setAlignment(Qt.AlignCenter)
        g_label.setStyleSheet("background-color:hsv(50, 211, 255)")

        h_label = QLabel('类型选择')
        h_label.setAlignment(Qt.AlignCenter)
        h_label.setStyleSheet("background-color:hsv(32, 248, 255)")

        i_label = QLabel('类型选择')
        i_label.setAlignment(Qt.AlignCenter)
        i_label.setStyleSheet("background-color:hsv(359, 253, 255)")

        # 创建水平方向滑动条
        rate_slider = QSlider(Qt.Horizontal)
        ##设置最小值
        rate_slider.setMinimum(0)
        # 设置最大值
        rate_slider.setMaximum(255)
        # 步长
        rate_slider.setSingleStep(5)
        # 设置当前值
        rate_slider.setValue(122)
        # 刻度位置，刻度下方
        rate_slider.setTickPosition(QSlider.TicksBelow)
        # 设置刻度间距
        rate_slider.setTickInterval(5)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(rate_slider)

        parameter_layout = QFormLayout()
        parameter_layout.addRow(all_label, select_all)
        parameter_layout.addRow(a_label, select_a)
        parameter_layout.addRow(b_label, select_b)
        parameter_layout.addRow(c_label, select_c)
        parameter_layout.addRow(d_label, select_d)
        parameter_layout.addRow(e_label, select_e)
        parameter_layout.addRow(f_label, select_f)
        parameter_layout.addRow(g_label, select_g)
        parameter_layout.addRow(h_label, select_h)
        parameter_layout.addRow(i_label, select_i)
        # parameter_layout.addRow("特征选择:", w)

        main_layout.addLayout(parameter_layout)
        main_layout.addLayout(vertical_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        ##全部添加到mainlayout
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        res = dialog.exec()
        select_type = []
        if res == QDialog.Accepted:
            if select_all.isChecked():
                for x in range(9):
                    select_type.append(x + 1)
            if select_a.isChecked():
                select_type.append(1)
            if select_b.isChecked():
                select_type.append(2)
            if select_c.isChecked():
                select_type.append(3)
            if select_d.isChecked():
                select_type.append(4)
            if select_e.isChecked():
                select_type.append(5)
            if select_f.isChecked():
                select_type.append(6)
            if select_g.isChecked():
                select_type.append(7)
            if select_h.isChecked():
                select_type.append(8)
            if select_i.isChecked():
                select_type.append(9)
            print(select_type)
            self.cancer_rate = int(rate_slider.value())
            self.clarity = 255
            self.tile_size = int(box_a.currentText())
            self.select_type = select_type
            self.slide_viewer.slide_graphics.update_grid_visibility(False)
            self.slide_viewer.cancer_type_show(self.select_type, self.clarity, self.cancer_rate, self.tile_size)
            self.slide_viewer.slide_graphics.update_grid_visibility(True)
        else:
            return
        return

    def set_alpha(self, clarity):
        self.clarity = clarity
        self.slide_viewer.slide_graphics.update_grid_visibility(False)
        self.slide_viewer.cancer_type_show(self.select_type, self.clarity, self.cancer_rate, self.tile_size)
        self.slide_viewer.slide_graphics.update_grid_visibility(True)

    def set_rate(self, cancer_rate):
        # print(self.select_type)
        self.cancer_rate = cancer_rate
        self.slide_viewer.slide_graphics.update_grid_visibility(False)
        self.slide_viewer.cancer_type_show(self.select_type, self.clarity, self.cancer_rate, self.tile_size)
        self.slide_viewer.slide_graphics.update_grid_visibility(True)

    def set_grid_size(self):
        dialog = QDialog()
        dialog.setWindowTitle("Grid size")

        main_layout = QVBoxLayout()

        box_a = QComboBox()
        box_a.addItem('Resnet-224')
        box_a.addItem('Tinynet-112')
        box_a.addItem('Tinynet-56')
        box_b = QComboBox()
        box_b.addItem('弱监督')
        box_b.addItem('非弱监督')
        box_layout_a = QHBoxLayout()
        box_layout_a.addWidget(box_a)
        box_layout_b = QHBoxLayout()
        box_layout_b.addWidget(box_b)

        box_c = QComboBox()
        box_c.addItem('32*32')
        box_c.addItem('56*56')
        box_layout_c = QHBoxLayout()
        box_layout_c.addWidget(box_c)

        box_d = QComboBox()
        box_d.addItem('滤波器')
        box_d.addItem('U-NET')
        box_layout_d = QHBoxLayout()
        box_layout_d.addWidget(box_d)

        box_e = QComboBox()
        box_e.addItem('20')
        box_e.addItem('40')
        box_layout_e = QHBoxLayout()
        box_layout_e.addWidget(box_e)

        box_f = QComboBox()
        box_f.addItem('50%')
        box_f.addItem('25%')
        box_f.addItem('75%')
        box_layout_f = QHBoxLayout()
        box_layout_f.addWidget(box_f)

        box_g = QComboBox()
        box_g.addItem('50%')
        box_g.addItem('25%')
        box_g.addItem('75%')
        box_layout_g = QHBoxLayout()
        box_layout_g.addWidget(box_g)

        box_final_layout = QFormLayout()
        box_final_layout.addRow("网络选取:", box_layout_a)
        box_final_layout.addRow("学习策略:", box_layout_b)
        box_final_layout.addRow("背滤尺寸:", box_layout_c)
        # box_final_layout.addRow("背滤方法:", box_layout_d)
        box_final_layout.addRow("选切层级:", box_layout_e)
        # box_final_layout.addRow("染色阈值:", box_layout_f)
        # box_final_layout.addRow("背滤阈值:", box_layout_g)

        main_layout.addLayout(box_final_layout)

        # magnification = QSpinBox()
        # magnification.setMaximum(80)
        # magnification.setMinimum(5)
        # magnification.setValue(20)
        #
        # ratio_a = QSpinBox()
        # ratio_a.setMaximum(100)
        # ratio_a.setMinimum(0)
        # ratio_a.setValue(50)
        #
        # ratio_b = QSpinBox()
        # ratio_b.setMaximum(255)
        # ratio_b.setMinimum(0)
        # ratio_b.setValue(50)
        #
        # vertical_layout = QVBoxLayout()
        # vertical_layout.addWidget(magnification)
        # vertical_layout.addWidget(ratio_a)
        # vertical_layout.addWidget(ratio_b)
        # parameter_layout = QFormLayout()
        # parameter_layout.addRow("选切层级:", magnification)
        # parameter_layout.addRow("归一阈值:", ratio_a)
        # parameter_layout.addRow("背滤阈值:", ratio_b)

        # main_layout.addLayout(parameter_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        ##全部添加到mainlayout
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        res = dialog.exec()
        distance = (0, 0)
        if res == QDialog.Accepted:
            slide_name = self.slide_viewer.slide_helper.slide_path
            slide_process = openslide.open_slide(slide_name)
            if box_a.currentText() == 'Resnet-224':
                tile_size = 224
            elif box_a.currentText() == 'Tinynet-112':
                tile_size = 112
            else:
                tile_size = 56

            # if box_b.currentText() == '弱监督':
            #     MIL_use = True
            # else:
            #     MIL_use = False
            MIL_use = True
            # if box_c.currentText() == '32*32':
            #     bin_size = 32
            # else:
            #     bin_size = 56
            bin_size = 32
            # if box_d.currentText() == 'U-NET':
            #     unet_use = True
            # else:
            unet_use = False
            self.bin_size = bin_size
            self.unet_use = unet_use
            self.tile_size = tile_size
            print('name', tile_size, "unetuse", unet_use)
            max_mag = slide_process.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            # slide.singleprocess_training_slides_to_images(slide_name)
            # print('slidedone')
            # filter.singleprocess_apply_filters_to_images()
            # print('filtersdone')
            # tiles.singleprocess_filtered_images_to_tiles()
            # print('tilesdone')
            # self.mythread = MyThread()  # 实例化自己建立的任务线程类
            # self.mythread.signal.connect(self.callback)  # 设置任务线程发射信号触发的函数
            # self.mythread.data = 5  # 这句就是给线程的实例化一个属性给其赋值，在线程里面就可以调用了
            # self.mythread.slidepath = slide_name
            # self.mythread.tilesize = tile_size
            # self.mythread.magnification = magnification.value()
            # self.mythread.ratio = float(ratio.value() / 100)
            # self.mythread.start()  # 启动任务线程

            pa = Process(target = nine_class_net,
                        args=(slide_name, tile_size, 20, float(50 / 100), MIL_use))
            pa.start()  ##多进程并行
            # if pa.is_alive():
            #     print('live')
            # else : print('kill')
            # t2 = time.time()
            # process_time = Manager().dict()
            # process_time['timeuse'] = time.time()
            # print("运行时间",process_time)
            pb = Process(target = background_filter, args = (slide_name, bin_size, unet_use))  ##注意进行操作时的传参格式加‘,’

            pb.start()  ##进程开始后在获取返回值
            # print('return', pb)

            # print('is _alive: ' + str(pb.is_alive()), pb)  #进程执行的状态判定
            grid_ratio = int(max_mag) / 20  ##不取int保持分数的可能性

            rects, color_alphas, cols, rows, cancer_type, mouse_rects_dict = build_rects_and_color_alphas_for_grid(
                (tile_size * grid_ratio, tile_size * grid_ratio),
                self.slide_viewer.slide_helper.get_level_size(0),
                distance
            )
            self.slide_viewer.slide_graphics.update_grid_rects_0_level(
                rects, color_alphas, distance, cancer_type, mouse_rects_dict, tile_size
            )
            return True
        else:
            return False

    def get_bin_param_trans(self):
        dialog = QDialog()
        dialog.setWindowTitle("backfilter size")
        main_layout = QVBoxLayout()
        box_a = QComboBox()
        box_a.addItem('32')
        box_a.addItem('56')
        box_b = QComboBox()

        box_b.addItem('滤波器')
        box_b.addItem('U-NET')
        box_layout = QHBoxLayout()
        box_layout.addWidget(box_a)
        box_layout.addWidget(box_b)
        box_final_layout = QFormLayout()
        box_final_layout.addRow("背滤尺寸:", box_layout)
        main_layout.addLayout(box_final_layout)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        ##全部添加到mainlayout
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        res = dialog.exec()

        if res == QDialog.Accepted:
            if box_b.currentText() == 'U-NET':
                unet_use = True
            else:
                unet_use = False
            self.unet_use = unet_use
            self.bin_size = int(box_a.currentText())
            self.slide_viewer.bin_show(self.bin_size, self.unet_use)
        else:
            return

    def get_net_param_trans(self):  # 向下传参

        dialog = QDialog()
        dialog.setWindowTitle("尺寸选择")
        main_layout = QVBoxLayout()
        box_a = QComboBox()
        box_a.addItem('224')
        box_a.addItem('112')
        box_a.addItem('56')
        box_layout = QHBoxLayout()
        box_layout.addWidget(box_a)
        box_final_layout = QFormLayout()
        box_final_layout.addRow("采样块尺寸:", box_layout)
        main_layout.addLayout(box_final_layout)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        ##全部添加到mainlayout
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        res = dialog.exec()
        if res == QDialog.Accepted:
            self.select_type = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.tile_size = int(box_a.currentText())

            self.slide_viewer.background_show(self.tile_size)

        else:
            return

    ##按键之后给他传参
    def show_rubber_band(self, checked):
        if (checked == True):
            rubber_can = 1
        else:
            rubber_can = 0
        self.slide_viewer.rubber_band_show(rubber_can)

    def show_grid(self, state: bool):
        # @pyqtSlot(bool)
        # def show_grid(self): #行不通    上面规定了state为bool类型参数
        self.slide_viewer.slide_graphics.update_grid_visibility(state)

    def go_to(self):
        dialog = QDialog()
        dialog.setWindowTitle("Location")

        level = MySpinBox(1)
        x = MySpinBox(1000)
        y = MySpinBox(1000)
        width = MySpinBox(1000)
        height = MySpinBox(1000)

        form_layout = QFormLayout()
        form_layout.addRow("level:", level)
        form_layout.addRow("x:", x)
        form_layout.addRow("y:", y)
        form_layout.addRow("width:", width)
        form_layout.addRow("height:", height)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        res = dialog.exec()
        if res == QDialog.Accepted:
            slide_path = self.slide_viewer.slide_helper.slide_path
            # qrectf = QRectF(x.value(), y.value(), width.value(), height.value())

            qrectf = (x.value(), y.value(), width.value(), height.value())
            self.slide_viewer.load(SlideViewParams(slide_path, level.value(), qrectf))
            # self.slide_viewer.load(SlideViewParams(slide_path, level.value()))

    def take_screenshot(self):
        dialog = QDialog()
        dialog.setWindowTitle("Screenshot")

        width = MySpinBox(1000)
        height = MySpinBox(1000)
        filepath = QLineEdit("screenshot_menu.jpg")

        form_layout = QFormLayout()
        form_layout.addRow("width:", width)
        form_layout.addRow("height:", height)
        form_layout.addRow("filepath:", filepath)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog
        )
        main_layout.addWidget(button_box)
        dialog.setLayout(main_layout)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        res = dialog.exec()
        if res == QDialog.Accepted:
            image = build_screenshot_image(
                self.slide_viewer.scene,
                QSize(width.value(), height.value()),
                self.slide_viewer.get_current_view_scene_rect(),

            )
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            str(now_time)
            image.save(filepath.text())

    def print_items(self):
        items = self.slide_viewer.scene.items(
            self.slide_viewer.get_current_view_scene_rect()
        )
        # print(items)
        QMessageBox.information(None, "Items", str(items))

    def print_slide_view_params(self):
        # str_ = to_json(self.slide_viewer.slide_view_params)
        # str_ = to_json(QRectF)(self.slide_viewer.slide_view_params)
        # print(str_)
        items = str('使用帮助：')
        QMessageBox.information(None, "SlideViewParams", items)

    'location 后print viewparam 报错出问题了' \
    '解决方案：装饰器问题添加print找到goto源头因为qrectf参数类型指定有问题现在改变传参类型解决' \
    '现在location刷新后需要点击一下才能刷新页底level'

    def seg_result_show(self):
        slide_path = self.slide_viewer.slide_helper.slide_path
        print(slide_path)
        # pc = Process(target=self.img_viewer.open(), args=(slide_path))  ##注意进行操作时的传参格式加‘,’
        self.dialog = img_viewed(slide_path)
        # self.dialog.show()
        self.dialog.exec()
        # pc.start()
        return



        #显示主窗体



# def mouse_track(mouse_ratio):
#         if (mouse_ratio == None):
#             return
#         else:
#             return mouse_ratio
