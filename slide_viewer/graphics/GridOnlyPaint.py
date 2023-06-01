from itertools import starmap
import numpy as np
from PyQt5.QtCore import QRectF, Qt, QRect
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget


class GridOnlyPaint(QGraphicsItem):
    def __init__(
            self,
            grid_rects_0_level: int,
            color_alphas,
            cancer_types,
            bounding_rect,
            select_type,
            rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9,
            clarity,
            base_color_rgb=(0, 0),  ##先设置了一个黑色色图层，控制a通道显示原图

    ):
        super().__init__()
        # if clarity == null:
        # print(clarity)
        self.grid_rects_0_level = grid_rects_0_level
        self.color_alphas = color_alphas
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setAcceptHoverEvents(False)
        self.bounding_rect = bounding_rect
        self.base_color_rgb = base_color_rgb
        self.cancer_types = cancer_types
        self.downsample = 1
        self.paint_black = 0
        self.star_map_ = starmap
        self.clarity = clarity
        self.cancer_rects = {}
        type_rate = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9]
        type = []
        self.cancertype = select_type[0]
        type.append(type_rate[self.cancertype-1])
        # print(len(type), a, type[0])

        self.color_alpha_rects_0_level = {}
        for color_alpha, grid_rect_0_level in zip(type[0], grid_rects_0_level):
            self.color_alpha_rects_0_level.setdefault(color_alpha, []).append(
                grid_rect_0_level
            )  ##setdefault 默认设置
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
        cancer_type = self.cancertype
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


        for color_alpha, rects in self.color_alpha_rects_0_level.items():
            color = QColor()
            if color_alpha == 0:
                color.setHsv(*self.base_color_rgb, color_alpha, 0)
            else:
                color.setHsv(*self.base_color_rgb, color_alpha, self.clarity)
            painter.setBrush(color)
            qrectfs = self.star_map_(QRectF, rects)
            painter.drawRects(qrectfs)  ##空指针

        painter.restore()



