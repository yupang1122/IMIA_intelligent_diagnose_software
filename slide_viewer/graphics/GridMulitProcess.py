from itertools import starmap

from PyQt5.QtCore import QRectF, Qt, QRect
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial


class GridMulitProcess(QGraphicsItem):
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
        self.downsample = 1
        self.paint_black = 0
        if clarity == None:
            clarity = 255
        self.star_map_ = starmap
        self.clarity = clarity
        self.color_alpha_rects_0_level = []
        self.cancer_rects = {}

        for x, y in zip(color_alphas, grid_rects_0_level):
            self.color_alpha_rects_0_level.append(
                [x, y]
            )

        for cancer_type, grid_rect_0_level in zip(cancer_types, self.color_alpha_rects_0_level):
            self.cancer_rects.setdefault(cancer_type, []).append(
                grid_rect_0_level
            )  ##进行分组设置
        self.recompute_bounding_rect()

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
                self.base_color_rgb = (255, 0)
            elif cancer_type == 2:
                self.base_color_rgb = (0, 255)
            elif cancer_type == 3:
                self.base_color_rgb = (0, 150)
            elif cancer_type == 4:
                self.base_color_rgb = (255, 90)
            elif cancer_type == 5:
                self.base_color_rgb = (0, 100)
            elif cancer_type == 6:
                self.base_color_rgb = (255, 200)
            elif cancer_type == 7:
                self.base_color_rgb = (120, 100)
            elif cancer_type == 8:
                self.base_color_rgb = (0, 50)
            elif cancer_type == 9:
                self.base_color_rgb = (0, 200)
            else:
                self.base_color_rgb = (0, 0)
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

            pool_param = [self.base_color_rgb, self.clarity]
            print('start')
            print(self.alpha_rects)
            pool_num = cpu_count() - 2
            t1 = time.time()
            pool = Pool(pool_num)
            partial_work = partial(paint_process, pool_param=pool_param)  # 偏函数打包
            pool.map(partial_work, self.alpha_rects.items())
            pool.close()
            pool.join()
            t2 = time.time()
            print("并行执行时间：", int(t2 - t1))
            print('end')

        #     for alpha, area in self.alpha_rects.items():  ##进行了标签分组不会遍历到每个数组元素
        #
        #         color = QColor()
        #         if alpha == 0:
        #             color.setHsv(*self.base_color_rgb,  alpha, 0)
        #         else:
        #             color.setHsv(*self.base_color_rgb, alpha, self.clarity)
        #         painter.setBrush(color)
        #         qrectfs = self.star_map_(QRectF, area)
        #         painter.drawRects(qrectfs)  ##空指针
        # painter.restore()


def paint_process(alpha_rects, pool_param):
    print('ok',pool_param, alpha_rects)
    base_color_rgb, clarity = pool_param
    painter = QPainter()
    for alpha, area in alpha_rects.items():  ##进行了标签分组不会遍历到每个数组元素
        print(alpha_rects.items())
        color = QColor()
        if alpha == 0:
            color.setHsv(*base_color_rgb, alpha, 0)
        else:
            color.setHsv(*base_color_rgb, alpha, clarity)
        painter.setBrush(color)
        qrectfs = starmap(QRectF, area)
        painter.drawRects(qrectfs)  ##空指针

    painter.restore()
