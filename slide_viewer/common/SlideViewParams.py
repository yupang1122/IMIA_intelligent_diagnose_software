from typing import List, Tuple

from slide_viewer.common.SlideHelper import SlideHelper


class SlideViewParams:
    def __init__(
        self,
        slide_path: str = None,
        level: int = None,
        level_rect: Tuple[float, float, float, float] = None,
        grid_rects_0_level: List[Tuple[float, float, float, float]] = None,
        grid_color_alphas_0_level: List[int] = None,
        grid_visible: bool = False,
        selected_rect_0_level: Tuple[float, float, float, float] = None,
        draw_line_0_level: List[int] = None,
        contours_0_level = None,
        init_level_and_level_rect_if_none=True,
        cancer_type: List[int] = None,
        select_type: List[int] = None,
        rate1: List[int] = None,
        rate2: List[int] = None,
        rate3: List[int] = None,
        rate4: List[int] = None,
        rate5: List[int] = None,
        rate6: List[int] = None,
        rate7: List[int] = None,
        rate8: List[int] = None,
        rate9: List[int] = None,
        mouse_rects_dict = dict(),
        tile_size = None,
        type_result = None,
        tiles_num = None,
        cell_num = None,
        tile_cell_num = None,
        rect_cell_result = [],
        area_mean = None,
        unet_use = None,
    ):
        super().__init__()
        self.slide_path = slide_path
        self.grid_rects_0_level = grid_rects_0_level
        self.grid_color_alphas_0_level = grid_color_alphas_0_level
        self.cancer_type = cancer_type
        self.select_type = select_type
        self.rate1 = rate1
        self.rate2 = rate2
        self.rate3 = rate3
        self.rate4 = rate4
        self.rate5 = rate5
        self.rate6 = rate6
        self.rate7 = rate7
        self.rate8 = rate8
        self.rate9 = rate9
        self.mouse_rects_dict = mouse_rects_dict
        self.tile_size = tile_size
        self.unet_use = unet_use
        self.grid_visible = grid_visible
        self.draw_line_0_level = draw_line_0_level
        self.selected_rect_0_level = selected_rect_0_level
        self.contours_0_level = contours_0_level

        if (level is None or level_rect is None) and init_level_and_level_rect_if_none:
            slide_helper = SlideHelper(slide_path)
            level = slide_helper.max_level
            level_rect = slide_helper.get_rect_for_level(level).getRect()

        self.level = level
        self.level_rect = level_rect

        self.type_result = type_result
        self.tiles_num = tiles_num
        self.cell_num = cell_num
        self.tile_cell_num = tile_cell_num
        self.rect_cell_result = rect_cell_result
        self.area_mean = area_mean

    def get_rect(self):
        # key = col + '_' + row
        # print(1)
        # print(self.mouse_rects_dict.get(key))
        # return self.mouse_rects_dict[key]
        return self.mouse_rects_dict

    @property
    def cache_key(self):
        cache_key = "{}_{}_{}_{}_{}_{}_{}".format(
            self.slide_path,
            self.level,
            self.level_rect,
            self.tile_size,
            self.grid_visible,
            self.cancer_type,
            self.select_type,
            self.unet_use,
            id(self.grid_rects_0_level),
            id(self.grid_color_alphas_0_level),
            id(self.rate1),
            id(self.rate2),
            id(self.rate3),
            id(self.rate4),
            id(self.rate5),
            id(self.rate6),
            id(self.rate7),
            id(self.rate8),
            id(self.rate9),
            id(self.draw_line_0_level),
            id(self.selected_rect_0_level),
            id(self.contours_0_level),
        )
        return cache_key
