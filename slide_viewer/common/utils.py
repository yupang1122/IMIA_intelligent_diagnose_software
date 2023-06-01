from math import ceil

from PyQt5.QtCore import QRectF, QPoint, QPointF
from PyQt5.QtGui import QPolygonF


def slice_rect(rect_size, tile_size):
    tiles_rects = []
    x, y = (0, 0)
    x_size, y_size = rect_size
    x_step, y_step = tile_size
    while y < y_size:
        while x < x_size:
            w = x_step
            if x + w >= x_size:
                w = x_size - x
            h = y_step
            if y + h >= y_size:
                h = y_size - y
            tiles_rects.append((x, y, w, h))
            x += x_step
        x = 0
        y += y_step
    return tiles_rects


def slice_rect2(rect_size, tile_size, tile_step):
    x_size, y_size = rect_size
    x_step, y_step = tile_step  ##步长
    cols = ceil(x_size / x_step)  ##列
    rows = ceil(y_size / y_step)  ##行
    rects = [
        (
            # (j - 1) * x_step + tile_size[0]+distance[0],
            # (i - 1) * y_step + tile_size[1]+distance[1],
            int((j - 1) * x_step + tile_size[0]),
            int((i - 1) * y_step + tile_size[1]),
            int(tile_size[0]),
            int(tile_size[1]),
        )
        for i in range(rows)
        for j in range(cols)
    ]

    if cols != x_size // x_step:
        for i in range(rows):
            rect = list(rects[i * cols + cols - 1])
            rect[2] = x_size - rect[0]
            rects[i * cols + cols - 1] = tuple(rect)
    if rows != y_size // y_step:
        for j in range(cols):
            rect = list(rects[(rows - 1) * cols + j])
            rect[3] = y_size - rect[1]
            rects[(rows - 1) * cols + j] = tuple(rect)

    # return rects
    return rects, cols, rows

def rect_to_str(rect):
    if isinstance(rect, QPolygonF):
        rect = rect.boundingRect()
    if isinstance(rect, QRectF):
        return "({:.2f}, {:.2f}, {:.2f}, {:.2f})".format(
            rect.x(), rect.y(), rect.bottomRight().x(), rect.bottomRight().y()
        )
    else:
        return "({}, {}, {}, {})".format(
            rect.x(), rect.y(), rect.bottomRight().x(), rect.bottomRight().y()
        )


def point_to_str(point: QPoint):
    if isinstance(point, QPointF):
        return "({:.2f}, {:.2f})".format(point.x(), point.y())
    else:
        return "({}, {})".format(point.x(), point.y())
