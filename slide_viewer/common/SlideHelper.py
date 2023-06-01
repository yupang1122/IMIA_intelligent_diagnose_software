from typing import List, Tuple
import numpy as np
import openslide
from openslide import deepzoom
from PyQt5.QtCore import QRectF


class SlideHelper:
    def __init__(self, slide_path: str):
        self._slide_path = slide_path
        with openslide.open_slide(slide_path) as slide:
            self.level_downsamples = slide.level_downsamples
            self.level_dimensions = slide.level_dimensions
            self.level_count = slide.level_count
            self.max_mag = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]

            magnification = 20
            tile_size = 224
            slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)  ##全部二倍下采样补充完整，并完成分割计算
            mag_num = np.log2(int(self.max_mag) / magnification) + 1  ##将分割的图像统一到同一尺度下进行算法处理
            if mag_num > 0:
                dz_level = int(slide_dz.level_count - mag_num)
            else:
                dz_level = int(slide_dz.level_count - 1)  ##找到下采样后对应层数    放大倍数magnification
            self.address = slide_dz.level_tiles[int(dz_level)]  # 找到对应层的分割行列数据 #为分割任务进行序号标记
            # print('helper', self.max_mag)

    @property
    def get_rects(self) -> int:
        return self.address

    @property
    def slide_path(self) -> str:
        return self._slide_path

    @property
    def max_level(self) -> int:
        return len(self.level_downsamples) - 1

    @property
    def get_best_mag_for_slide(self) -> int:
        grid_ratio = int(self.max_mag) / 20
        grid_size = int(224 * grid_ratio)
        mag = 20
        return grid_ratio, grid_size, mag   ##保持小数可能性

    @property
    def levels(self) -> List[int]:
        return list(range(self.level_count))

    def get_downsample_for_level(self, level: int) -> float:
        return self.level_downsamples[level]

    def get_level_size(self, level: int) -> Tuple[int, int]:
        return self.level_dimensions[level]

    def get_rect_for_level(self, level: int) -> QRectF:
        size = self.get_level_size(level)
        rect = QRectF(0, 0, size[0], size[1])
        return rect

    def get_best_level_for_downsample(self, downsample: float) -> int:
        with openslide.open_slide(self._slide_path) as slide:
            # print(slide.get_best_level_for_downsample(downsample))
            return slide.get_best_level_for_downsample(downsample)


