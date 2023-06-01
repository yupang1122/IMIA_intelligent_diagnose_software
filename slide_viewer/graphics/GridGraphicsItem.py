from itertools import starmap

from PyQt5.QtCore import QRectF, Qt, QRect
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget


class GridGraphicsItem(QGraphicsItem):
    def __init__(
            self,
            grid_rects_0_level: int,
            color_alphas,
            cancer_types,
            bounding_rect,
            clarity=None,
            base_color_rgb=(0, 0),  ##先设置了一个黑色色图层，控制a通道显示原图

    ):
        super().__init__()
        self.grid_rects_0_level = grid_rects_0_level
        self.color_alphas = color_alphas
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setAcceptHoverEvents(False)
        self.bounding_rect = bounding_rect
        self.base_color_rgb = base_color_rgb
        self.cancer_types = cancer_types
        # print('grid', cancer_types)
        # print(color_alphas)
        # print(grid_rects_0_level)
        # self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)
        self.downsample = 1
        self.paint_black = 0
        # print('clarity', clarity)
        if clarity == None:
            clarity = 255
        self.star_map_ = starmap
        self.clarity = clarity
        self.color_alpha_rects_0_level = []
        self.cancer_rects = {}
        # zip对区域和a值打包一一对应
        for x, y in zip(color_alphas, grid_rects_0_level):
            self.color_alpha_rects_0_level.append(
                [x, y]
            )
        # print(self.color_alpha_rects_0_level)

        for cancer_type, grid_rect_0_level in zip(cancer_types, self.color_alpha_rects_0_level):
            self.cancer_rects.setdefault(cancer_type, []).append(
                grid_rect_0_level
            )  ##进行分组设置
            ##setdefault 默认设置
            # print('grid',grid_rect_0_level)
        # print(self.cancer_rects)
        # print(self.cancer_rects.items())  ##字典化
        # for color_alpha, rects in self.color_alpha_rects_0_level.items():
        #     print(color_alpha)

        self.recompute_bounding_rect()

        # print(len(color_alphas), color_alphas)
        # print(len(grid_rects_0_level), grid_rects_0_level)
        # for color_alpha, grid_rect_0_level in zip(color_alphas, grid_rects_0_level):
        #     # self.color_alpha_rects_0_level[color_alpha] = grid_rect_0_level  ##setdefault 默认设置
        #     self.color_alpha_rects_0_level.setdefault(color_alpha, []).append(
        #                 grid_rect_0_level
        #             )   ##setdefault 默认设置
        #     # print('grid',grid_rect_0_level)
        # # print(grid_rects_0_level)\
        #     print(color_alpha,grid_rect_0_level)
        #     print(self.color_alpha_rects_0_level)
        # print(len(self.color_alpha_rects_0_level))
        # self.recompute_bounding_rect()
        # return self.color_alpha_rects_0_level

    def recompute_bounding_rect(self):
        self.bounding_qrectf = QRectF(
            self.bounding_rect[0],
            self.bounding_rect[1],
            self.bounding_rect[2] / self.downsample,
            self.bounding_rect[3] / self.downsample,
        )

    def update_downsample(self, downsample):
        self.downsample = downsample
        self.recompute_bounding_rect()

    def boundingRect(self):
        return self.bounding_qrectf

    def paint(
            self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    ):
        painter.save()
        scale = 1 / self.downsample
        painter.scale(scale, scale)
        for cancer_type, rects in self.cancer_rects.items():
            if cancer_type == 1:
                self.base_color_rgb = (239, 244)
            elif cancer_type == 2:
                self.base_color_rgb = (303, 129)
            elif cancer_type == 3:
                self.base_color_rgb = (272, 252)
            elif cancer_type == 4:
                self.base_color_rgb = (217, 106)
            elif cancer_type == 5:
                self.base_color_rgb = (168, 223)
            elif cancer_type == 6:
                self.base_color_rgb = (120, 253)
            elif cancer_type == 7:
                self.base_color_rgb = (50, 211)
            elif cancer_type == 8:
                self.base_color_rgb = (32, 248)
            elif cancer_type == 9:
                self.base_color_rgb = (359, 253)
            else:
                self.base_color_rgb = (0, 0)
            # if cancer_type == 0:
            #     self.base_color_rgb = (0, 0)
            # else:
            #     self.base_color_rgb = (158, 150)
            a = []
            b = []

            for rect in rects:
                a.append(rect[1])
                b.append(rect[0])
            self.alpha_rects = {}
            for x, y in zip(a, b):
                self.alpha_rects.setdefault(y, []).append(
                    x
                )

            #
            # pool_param = [self.base_color_rgb, self.clarity]
            # print('start')
            # pool_num = cpu_count() - 2
            # t1 = time.time()
            # pool = Pool(pool_num)
            #
            # partial_work = partial(self.paint_process, pool_param=pool_param)  # 偏函数打包
            # pool.map(partial_work, self.alpha_rects.items())
            # pool.close()
            # pool.join()
            # t2 = time.time()
            # print("并行执行时间：", int(t2 - t1))
            # print('end')
        # #
            for alpha, area in self.alpha_rects.items():  ##进行了标签分组不会遍历到每个数组元素
                #alpha 为色深控制参数   clarity 控制透明度
                color = QColor()
                if alpha == 0:
                    color.setHsv(*self.base_color_rgb,  alpha, 0)
                else:
                    color.setHsv(*self.base_color_rgb, alpha, self.clarity)
                painter.setBrush(color)
                qrectfs = self.star_map_(QRectF, area)
                painter.drawRects(qrectfs)  ##空指针
        painter.restore()

    #
    # def paint_process(self, alpha_rects):
    #     painter = QPainter()
    #     for alpha, area in alpha_rects.items():  ##进行了标签分组不会遍历到每个数组元素
    #         print(alpha_rects.items())
    #         print(alpha)
    #         color = QColor()
    #         if alpha == 0:
    #             color.setHsv(*self.base_color_rgb, alpha, 0)
    #         else:
    #             color.setHsv(*self.base_color_rgb, alpha, self.clarity)
    #         painter.setBrush(color)
    #         qrectfs = starmap(QRectF, area)
    #         painter.drawRects(qrectfs)  ##空指针
    #
    #     painter.restore()

    # def paint(
    #     self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    # ):
    #     painter.save()
    #     scale = 1 / self.downsample
    #     painter.scale(scale, scale)
    #     ##分析这个循环
    #     for color_alpha, rects in self.color_alpha_rects_0_level.items():
    #         # color = QColor(*self.base_color_rgb, color_alpha)
    #         #在此处增加对color变量的案件判断 将每个不同的alpha值对应的区域各自整理成集合去绘制
    #         if 0 <= color_alpha <= 51 :
    #             self.base_color_rgb = (255,0,0)
    #         elif 51 < color_alpha <= 105:
    #             self.base_color_rgb = (0,255,0)
    #         elif 105 < color_alpha <= 160:
    #             self.base_color_rgb = (0,0,255)
    #         elif 160 < color_alpha < 255:
    #             self.base_color_rgb = (255,125,0)
    #         else:
    #             self.base_color_rgb = (0,0,0)
    #           # print('draw',color_alpha,rects)
    #         color = QColor(*self.base_color_rgb, color_alpha)
    #         # print(color)
    #         # painter.setBrush(QColor(0, 0, 0, 0))      ##网格内块填充色 COLOR可能为空指针没有意义没有数值
    #         painter.setBrush(color)
    #         # for rect in rects:
    #         #     pick = rect[0]
    #         #     print(pick)
    #         # print(rects)
    #         qrectfs = self.star_map_(QRectF, rects)
    #         # print(qrectfs)
    #         painter.drawRects(qrectfs)     ##空指针
    #     painter.restore()
    #     # print('paint grid')
    #     ##类似一个无定义鼠标事件变量为空

    # def paint_background(
    #     self,
    # ):
    #     if self.paint_black == 1 :
    #         painter = QPainter()
    #         painter.save()
    #         scale = 1 / self.downsample
    #         painter.scale(scale, scale)
    #         ##分析这个循环
    #         for color_alpha, rects in self.color_alpha_rects_0_level.items():
    #             # color = QColor(*self.base_color_rgb, color_alpha)
    #             color = QColor(*self.base_color_rgb, 120)
    #             # print(color)
    #             # painter.setBrush(QColor(0, 0, 0, 0))      ##网格内块填充色 COLOR可能为空指针没有意义没有数值
    #             painter.setBrush(color)
    #             qrectfs = self.star_map_(QRectF, rects)
    #             # print(qrectfs)
    #             painter.drawRects(qrectfs)  ##空指针
    #         painter.restore()
    #     else :
    #         #将参数传到这一层方便计算坐标
    #         return
