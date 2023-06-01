from itertools import starmap
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget


class GridBinfilterPaint(QGraphicsItem):
    def __init__(
            self,
            grid_rects_0_level: int,
            color_alphas,
            unet_use,
            bounding_rect,

            base_color_rgb=(0, 0, 0),  ##先设置了一个黑色色图层，控制a通道显示原图
            clarity=255
    ):
        super().__init__()
        self.grid_rects_0_level = grid_rects_0_level
        self.color_alphas = color_alphas
        self.unet_use = unet_use
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setAcceptHoverEvents(False)
        self.bounding_rect = bounding_rect
        if unet_use:
            self.base_color_rgb = base_color_rgb
        else:
            self.base_color_rgb = (69,127,49)
        self.downsample = 1

        self.star_map_ = starmap

        self.grid_rects = {}
        # zip对区域和a值打包一一对应

        for color_alpha, grid_rect_0_level in zip(color_alphas, grid_rects_0_level):
            self.grid_rects.setdefault(color_alpha, []).append(
                grid_rect_0_level
            )  ##进行分组设置
        # print('filterpaint', self.grid_rects)
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

        for color_alpha, rects in self.grid_rects.items():
            color = QColor()
            # print(len(rects),color_alpha)
            color.setRgb(*self.base_color_rgb, color_alpha)
            painter.setBrush(color)
            qrectfs = self.star_map_(QRectF, rects)
            painter.drawRects(qrectfs)  ##空指针
        painter.restore()
