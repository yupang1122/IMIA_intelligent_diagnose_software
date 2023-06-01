from PyQt5.QtCore import QRectF, Qt, QLine, QPoint
from PyQt5.QtWidgets import *

from slide_viewer.common.level_builders import (
    build_tiles_level,
    build_grid_level_from_rects,
    build_type_select_from_rects,
    build_bin_level_from_rects,
)
from slide_viewer.common.SlideHelper import SlideHelper
from slide_viewer.common.SlideViewParams import SlideViewParams
from slide_viewer.graphics.LeveledGraphicsItemGroup import LeveledGraphicsItemGroup
from slide_viewer.graphics.SelectedRectGraphicsItem import SelectedRectGraphicsItem
from slide_viewer.graphics.PickRectGraphicsItem import PickRectGraphicsItem
from slide_viewer.graphics.DrawLineGraphicsItem import DrawLineGraphicsItem
from slide_viewer.graphics.PainContoursGraphicsItem import PainContoursGraphicsItem

class SlideGraphicsItemGroup(QGraphicsItemGroup):
    def __init__(self, slide_view_params: SlideViewParams, distance, preffered_rects_count=2000):
        super().__init__()
        self.slide_view_params = slide_view_params
        self.slide_helper = SlideHelper(slide_view_params.slide_path)

        slide_w, slide_h = self.slide_helper.get_level_size(0)
        t = ((slide_w * slide_h) / preffered_rects_count) ** 0.5
        if t < 1000:

            t = 1000

        self.tile_size = (int(t), int(t))

        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setAcceptHoverEvents(False)

        self.levels = self.slide_helper.levels

        self.leveled_graphics_group = LeveledGraphicsItemGroup(self.levels, self)
        self.leveled_graphics_selection = LeveledGraphicsItemGroup(self.levels, self)
        self.leveled_groups = [
            self.leveled_graphics_group,
            self.leveled_graphics_selection,
        ]

        self.graphics_grid = None

        # print(slide_view_params.slide_path)
        # print("=" * 100)
        # with elapsed_timer() as elapsed:
        self.init_tiles_levels()
        # print("init_tiles_levels", elapsed())
        self.init_grid_levels(distance)
        self.init_bin_levels()
        # self.init_type_select(distance)
        # print("init_grid_levels", elapsed())
        self.init_selected_rect_levels()
        self.update_visible_level(self.slide_view_params.level)


    def boundingRect(self) -> QRectF:
        return self.leveled_graphics_group.boundingRect()


    ##图元置入scene的初始化
    def init_tiles_levels(self):
        for level in self.levels:
            tiles_level = build_tiles_level(level, self.tile_size, self.slide_helper)
            self.leveled_graphics_group.clear_level(level)
            self.leveled_graphics_group.add_item_to_level_group(level, tiles_level)

    def init_bin_levels(self):
        if self.slide_view_params.grid_rects_0_level:
            level = 0
            graphics_grid = build_bin_level_from_rects(
                level,
                self.slide_view_params.grid_rects_0_level,  ##已经完成关于鼠标拖动距离的创建
                self.slide_view_params.grid_color_alphas_0_level,  ###生成a值
                self.slide_view_params.unet_use,
                self.slide_helper,
            )
            graphics_grid.setZValue(10)  ##定在最高层
            graphics_grid.setVisible(self.slide_view_params.grid_visible)
            self.addToGroup(graphics_grid)
            self.graphics_grid = graphics_grid

    def init_grid_levels(self, distance):
        if self.slide_view_params.grid_rects_0_level:
            level = 0
            #绘制的mask网格变量
            graphics_grid = build_grid_level_from_rects(
                level,
                self.slide_view_params.grid_rects_0_level, ##已经完成关于鼠标拖动距离的创建
                self.slide_view_params.grid_color_alphas_0_level, ###生成a值
                self.slide_view_params.cancer_type, ##患癌类型
                distance,
                self.slide_helper,
            )
            # graphics_grid.setZValue(10)
            graphics_grid.setZValue(10)  ##定在最高层
            graphics_grid.setVisible(self.slide_view_params.grid_visible)
            # print('group',graphics_grid)
            self.addToGroup(graphics_grid)
            self.graphics_grid = graphics_grid

    def init_type_select(self, clarity):
        if self.slide_view_params.grid_rects_0_level:
            level = 0
            graphics_grid = build_type_select_from_rects(
                level,
                self.slide_view_params.grid_rects_0_level, ##已经完成关于鼠标拖动距离的创建
                self.slide_view_params.grid_color_alphas_0_level, ###生成a值
                self.slide_view_params.cancer_type, ##患癌类型
                self.slide_view_params.select_type,
                self.slide_view_params.rate1,
                self.slide_view_params.rate2,
                self.slide_view_params.rate3,
                self.slide_view_params.rate4,
                self.slide_view_params.rate5,
                self.slide_view_params.rate6,
                self.slide_view_params.rate7,
                self.slide_view_params.rate8,
                self.slide_view_params.rate9,
                clarity,
                self.slide_helper,
            )
            # graphics_grid.setZValue(10)
            graphics_grid.setZValue(10)  ##定在最高层
            graphics_grid.setVisible(self.slide_view_params.grid_visible)
            # print('group',graphics_grid)
            self.addToGroup(graphics_grid)
            self.graphics_grid = graphics_grid

    def init_selected_rect_levels(self):
        if self.slide_view_params.selected_rect_0_level:
            for level in self.levels:             ##level的循环对每一层都进行计算
                downsample = self.slide_helper.get_downsample_for_level(level)
                selected_qrectf_0_level = QRectF(
                    *self.slide_view_params.selected_rect_0_level
                )
                rect_for_level = QRectF(
                    selected_qrectf_0_level.topLeft() / downsample,
                    selected_qrectf_0_level.size() / downsample,
                )
                # selected_graphics_rect = SelectedRectGraphicsItem(rect_for_level)     ##获取到每一个层级下的框选区域
                selected_graphics_rect = SelectedRectGraphicsItem(rect_for_level, level)
                selected_graphics_rect.setZValue(20)     ###规定层级保持选框在最高层
                self.leveled_graphics_selection.clear_level(level)
                self.leveled_graphics_selection.add_item_to_level_group(
                    level, selected_graphics_rect
                )

    def init_contours_levels(self):
        if self.slide_view_params.contours_0_level:
            for level in self.levels:
                downsample = self.slide_helper.get_downsample_for_level(level)
                contours_for_level = []
                for i in range(len(self.slide_view_params.contours_0_level)):
                    a = QPoint(self.slide_view_params.contours_0_level[i][0] / downsample,
                                                self.slide_view_params.contours_0_level[i][1] / downsample)
                    contours_for_level.append(a)
                selected_qrectf_0_level = QRectF(
                    *self.slide_view_params.selected_rect_0_level
                )
                rect_for_level = QRectF(
                    selected_qrectf_0_level.topLeft() / downsample,
                    selected_qrectf_0_level.size() / downsample,
                )
                contours_graphics_rect = PainContoursGraphicsItem(contours_for_level, level, rect_for_level)
                contours_graphics_rect.setZValue(20)     ###规定层级保持选框在最高层
                # self.leveled_graphics_selection.clear_level(level)
                self.leveled_graphics_selection.add_item_to_level_group(
                    level, contours_graphics_rect
                )



    def init_draw_line_levels(self):
        if self.slide_view_params.draw_line_0_level:
            for level in self.levels:             ##level的循环对每一层都进行计算
                downsample = self.slide_helper.get_downsample_for_level(level)
                # print(self.slide_view_params.draw_line_0_level,'param')
                a,b,c,d = self.slide_view_params.draw_line_0_level
                # print(a,b,c,d)
                line_for_level = QLine(a/ downsample,b/ downsample,c/ downsample,d/ downsample)
                # print(line_for_level)
                selected_graphics_rect = DrawLineGraphicsItem(line_for_level, level, a/ downsample, b/ downsample)
                selected_graphics_rect.setZValue(20)     ###规定层级保持选框在最高层
                # self.leveled_graphics_selection.clear_level(level)
                self.leveled_graphics_selection.add_item_to_level_group(
                    level, selected_graphics_rect
                )

    def init_pick_rect_levels(self):
        if self.slide_view_params.selected_rect_0_level:
            for level in self.levels:             ##level的循环对每一层都进行计算
                downsample = self.slide_helper.get_downsample_for_level(level)
                selected_qrectf_0_level = QRectF(
                    *self.slide_view_params.selected_rect_0_level
                )
                print('param',selected_qrectf_0_level,self.slide_view_params.selected_rect_0_level)
                rect_for_level = QRectF(
                    selected_qrectf_0_level.topLeft() / downsample,
                    selected_qrectf_0_level.size() / downsample,
                ) ##左上和size
                # selected_graphics_rect = SelectedRectGraphicsItem(rect_for_level,level)     ##获取到每一个层级下的框选区域
                selected_graphics_rect = PickRectGraphicsItem(rect_for_level, level)
                selected_graphics_rect.setZValue(20)     ###规定层级保持选框在最高层
                self.leveled_graphics_selection.clear_level(level)  ##控制并列选框
                self.leveled_graphics_selection.add_item_to_level_group(
                    level, selected_graphics_rect
                )

    def init_click_rect_levels(self):
        if self.slide_view_params.selected_rect_0_level:
            for level in self.levels:             ##level的循环对每一层都进行计算
                downsample = self.slide_helper.get_downsample_for_level(level)
                selected_qrectf_0_level = QRectF(
                    *self.slide_view_params.selected_rect_0_level
                )
                rect_for_level = QRectF(
                    selected_qrectf_0_level.topLeft() / downsample,
                    selected_qrectf_0_level.size() / downsample,
                ) ##左上和size
                selected_graphics_rect = SelectedRectGraphicsItem(rect_for_level, level)     ##获取到每一个层级下的框选区域
                # selected_graphics_rect = PickRectGraphicsItem(rect_for_level, level)
                selected_graphics_rect.setZValue(20)     ###规定层级保持选框在最高层
                # self.leveled_graphics_selection.clear_level(level)  ##控制并列选框  保证始终只有一个
                self.leveled_graphics_selection.add_item_to_level_group(
                    level, selected_graphics_rect
                )

    def update_visible_level(self, visible_level):
        if visible_level == None or visible_level == -1:
            visible_level = max(self.levels)
        for leveled_group in self.leveled_groups:
            leveled_group.update_visible_level(visible_level)
        self.slide_view_params.level = visible_level


        if self.graphics_grid:
            self.graphics_grid.update_downsample(
                self.slide_helper.get_downsample_for_level(visible_level)
            )

    #参数整合函数的数据存入
    def update_bin_rects_0_level(self, grid_rects_0_level, grid_color_alphas_0_level, tile_size, unet_use):
        self.slide_view_params.grid_rects_0_level = grid_rects_0_level  ##为param赋值
        self.slide_view_params.grid_color_alphas_0_level = grid_color_alphas_0_level  ##a通道值  保持透明为零
        self.slide_view_params.tile_size = tile_size
        self.slide_view_params.unet_use = unet_use

        self.init_bin_levels()

    def update_grid_rects_0_level(self, grid_rects_0_level, grid_color_alphas_0_level, distance, cancer_type, mouse_rects_dict, tile_size):
        self.slide_view_params.grid_rects_0_level = grid_rects_0_level  ##为param赋值
        self.slide_view_params.grid_color_alphas_0_level = grid_color_alphas_0_level  ##a通道值  保持透明为零
        self.slide_view_params.cancer_type = cancer_type
        self.slide_view_params.mouse_rects_dict = mouse_rects_dict
        self.slide_view_params.tile_size = tile_size
        # print(grid_color_alphas_0_level)
        self.init_grid_levels(distance)

    def update_grid_rects_type_select(self, grid_rects_0_level, grid_color_alphas_0_level, distance, cancer_type, select_type,
                                      rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, clarity, mouse_rects_dict, tile_size, tiles_num):
        self.slide_view_params.grid_rects_0_level = grid_rects_0_level  ##为param赋值
        self.slide_view_params.grid_color_alphas_0_level = grid_color_alphas_0_level  ##a通道值  保持透明为零
        self.slide_view_params.cancer_type = cancer_type
        self.slide_view_params.select_type = select_type
        self.slide_view_params.rate1 = rate1
        self.slide_view_params.rate2 = rate2
        self.slide_view_params.rate3 = rate3
        self.slide_view_params.rate4 = rate4
        self.slide_view_params.rate5 = rate5
        self.slide_view_params.rate6 = rate6
        self.slide_view_params.rate7 = rate7
        self.slide_view_params.rate8 = rate8
        self.slide_view_params.rate9 = rate9
        self.slide_view_params.mouse_rects_dict = mouse_rects_dict
        self.slide_view_params.tile_size = tile_size
        self.slide_view_params.tiles_num = tiles_num
        # print(grid_color_alphas_0_level)
        self.init_type_select(clarity)

    def update_grid_visibility(self, grid_visible):
        self.slide_view_params.grid_visible = grid_visible
        if self.graphics_grid:
            self.graphics_grid.update_downsample(
                self.slide_helper.get_downsample_for_level(self.slide_view_params.level)
            )
            self.graphics_grid.setVisible(grid_visible)

    def update_selected_rect_0_level(self, selected_rect_0_level):
        self.slide_view_params.selected_rect_0_level = selected_rect_0_level
        self.init_selected_rect_levels()

    def update_pick_rect_0_level(self, selected_rect_0_level):
        self.slide_view_params.selected_rect_0_level = selected_rect_0_level
        self.init_pick_rect_levels()

    def update_click_rect_0_level(self, selected_rect_0_level):
        self.slide_view_params.selected_rect_0_level = selected_rect_0_level
        self.init_click_rect_levels()

    def update_draw_line_0_level(self, draw_line_0_level):
        self.slide_view_params.draw_line_0_level = draw_line_0_level
        self.init_draw_line_levels()

    def update_paint_contours_0_level(self, contours_0_level):
        self.slide_view_params.contours_0_level = contours_0_level
        self.init_contours_levels()

    def clear_level(self):
        for level in self.levels:
            self.leveled_graphics_selection.clear_level(level)