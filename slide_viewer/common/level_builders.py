from PyQt5.QtWidgets import QGraphicsItemGroup
import numpy as np
from slide_viewer.common.SlideHelper import SlideHelper
from slide_viewer.common.utils import slice_rect, slice_rect2
from slide_viewer.graphics.GridGraphicsItem import GridGraphicsItem
from slide_viewer.graphics.GridOnlyPaint import GridOnlyPaint
from slide_viewer.graphics.GridAlpha0Paint import GridAlpha0Paint
from slide_viewer.graphics.GridMulitProcess import GridMulitProcess
from slide_viewer.graphics.TileGraphicsItem import TileGraphicsItem
from slide_viewer.graphics.GridBinfilterPaint import GridBinfilterPaint
import random
import os
import operator


def build_tiles_level(level, tile_size, slide_helper: SlideHelper):
    level_size = slide_helper.get_level_size(level)
    tiles_rects = slice_rect(level_size, tile_size)
    tiles_graphics_group = QGraphicsItemGroup()
    downsample = slide_helper.get_downsample_for_level(level)
    for tile_rect in tiles_rects:
        item = TileGraphicsItem(tile_rect, slide_helper.slide_path, level, downsample)
        item.moveBy(tile_rect[0], tile_rect[1])
        tiles_graphics_group.addToGroup(item)

    return tiles_graphics_group


def build_rects_and_color_alphas_for_grid(
        grid_size_0_level, level_size_0, distance, slice_func=slice_rect2
):
    rect_size = grid_size_0_level[0], grid_size_0_level[1]
    rects, cols, rows = slice_func(level_size_0, rect_size, rect_size)
    # color_alphas = [(0, 255, 0, random.randint(0, 128)) for i in range(len(rects))]
    # color_alphas = [(0, 255, 0, 0) for i in range(len(rects))]
    color_alphas = [0 for i in range(len(rects))]  ##无值闪退  有值为零  .rects为每一个小块坐标定为零保持显示完整性
    # print(rects, 'builders')   ##rect为画好的区块

    cancer_type = [0 for i in range(len(rects))]
    mouse_rects_dict = dict()
    return rects, color_alphas, cols, rows, cancer_type, mouse_rects_dict

def build_rects_and_color_alphas_for_filter(grid_size_0_level, level_size_0, slide_name, bin_size, slice_func = slice_rect2):
    svs_name = slide_name.split('/')[-1][:-4]
    bin_path = os.path.join('../dataprocess/data-index', svs_name)
    bin_index = bin_path + '/' + 'back_index.txt'
    rect_size = grid_size_0_level[0], grid_size_0_level[1]
    rects, cols, rows = slice_func(level_size_0, rect_size, rect_size)  ##向内传参
    print('filtercolrow', cols,rows, len(rects))
    color_alphas = []
    with open(bin_index,  'r') as f:
        annotations = f.readlines()
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        # print(annotation)
        annotation = list(map(int,annotation))
        color_alphas.extend(annotation)
        if len(annotation) != cols:
            print('pip_fail', len(annotation))
        else:
            continue
    # print(color_alphas)
    print(rects[0],rects[1], rects[2])
    print(len(color_alphas),'filteralpha')
    return rects, cols, rows, color_alphas

def build_rects_and_color_alphas_for_background(
        grid_size_0_level, level_size_0, slide_name, distance, tile_size, slice_func=slice_rect2
):
    svs_name = slide_name.split('/')[-1][:-4]
    index_path = os.path.join('../dataprocess/data-index/', svs_name, str(tile_size))  ##背景筛选信息位置 创建这个文件夹
    background_index = index_path + '/' + 'svs_index.txt'
    rect_size = grid_size_0_level[0], grid_size_0_level[1]
    rects, cols, rows = slice_func(level_size_0, rect_size, rect_size)  ##向内传参
    with open(background_index, 'r') as f:
        annotations = f.readlines()
    color_alphas = []
    cancer_type = []
    # print(cols, rows, rects)
    area = np.array(list(rects)).reshape(rows, cols, 4)
    # print(area)
    fix_area = []
    ##如果要进行数组间一一对应则无法用多线程推线程池的方法加速
    mouse_rects_dict = dict()
    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        row = int(annotation[3])
        col = int(annotation[2])
        fix_area.append(area[row][col])
        max_rate = int(annotation[1])
        color_alphas.append(max_rate)
        max_type = int(annotation[4])
        cancer_type.append(max_type)
        key = str(col) + '_' + str(row)
        rects_dict = {key: [max_rate, max_type]}
        mouse_rects_dict.update(rects_dict)
        continue
    # print(fix_area)
    # for row in range(rows):
    #     for col in range(cols):
    #         for annotation in annotations:
    #             annotation = annotation.strip().split(' ')
    #             if int(row) == int(annotation[3]):
    #                 if int(col) == int(annotation[2]):
    #                     # print('anno', annotation[2], annotation[3])
    #                     # if int(annotation[1]) == 255:
    #                     #     background_choose = int(annotation[1])
    #                     # else:
    #                     #     background_choose = int(np.random.uniform(0,1) * 255 )
    #                     # print(background_choose)
    #                     background_choose = int(annotation[1])
    #                     color_alphas.append(background_choose)
    #                     cancer_type.append(int(annotation[4]))
    #                 else:
    #                     continue
    #             else:
    #                 continue
    #     continue
    print(color_alphas)
    return fix_area, color_alphas, cols, rows, cancer_type, mouse_rects_dict

    # pool_param = [slide_name, tile_size, ratio, out_path, index_path, dz_level, address]
    # print('start')
    # pool_num = cpu_count() - 2
    # t1 = time.time()
    # pool = Pool(pool_num)
    # row = []
    # for i in range(address[1]):
    #     row.append(i)
    #     continue
    # partial_work = partial(pool_process, pool_param=pool_param)  # 偏函数打包
    # pool.map(partial_work, row)
    # pool.close()
    # pool.join()
    # t2 = time.time()
    # print("并行执行时间：", int(t2 - t1))


def build_rects_and_color_alphas_for_type_select(
        grid_size_0_level, level_size_0, slide_name, distance, cancer_rate, tile_size, slice_func=slice_rect2
):
    svs_name = slide_name.split('/')[-1][:-4]
    index_path = os.path.join('../dataprocess/data-index/', svs_name, str(tile_size))  ##背景筛选信息位置 创建这个文件夹
    background_index = index_path + '/' + 'svs_index.txt'
    rect_size = grid_size_0_level[0], grid_size_0_level[1]
    rects, cols, rows = slice_func(level_size_0, rect_size, rect_size)  ##向内传参
    with open(background_index, 'r') as f:
        annotations = f.readlines()
    color_alphas = []
    cancer_type = []
    rate1 = []
    rate2 = []
    rate3 = []
    rate4 = []
    rate5 = []
    rate6 = []
    rate7 = []
    rate8 = []
    rate9 = []

    area = np.array(list(rects)).reshape(rows, cols, 4)
    # print(area)
    fix_area = []
    ##如果要进行数组间一一对应则无法用多线程推线程池的方法加速

    mouse_rects_dict = dict()
    for annotation in annotations:
        annotation = annotation.strip().split(' ')  #去掉空格
        row = int(annotation[3])
        col = int(annotation[2])
        fix_area.append(area[row][col])
        max_rate = int(annotation[1])
        color_alphas.append(max_rate)
        max_type = int(annotation[4])
        cancer_type.append(max_type)
        key = str(col) + '_' + str(row)
        rects_dict = {key: [max_rate, max_type]}
        mouse_rects_dict.update(rects_dict)

        if int(cancer_rate) < int(annotation[5]):
            rate1.append(int(annotation[5]))
        else:
            rate1.append(0)
        if int(cancer_rate) < int(annotation[6]):
            rate2.append(int(annotation[6]))
        else:
            rate2.append(0)
        if int(cancer_rate) < int(annotation[7]):
            rate3.append(int(annotation[7]))
        else:
            rate3.append(0)
        if int(cancer_rate) < int(annotation[8]):
            rate4.append(int(annotation[8]))
        else:
            rate4.append(0)
        if int(cancer_rate) < int(annotation[9]):
            rate5.append(int(annotation[9]))
        else:
            rate5.append(0)
        if int(cancer_rate) < int(annotation[10]):
            rate6.append(int(annotation[10]))
        else:
            rate6.append(0)
        if int(cancer_rate) < int(annotation[11]):
            rate7.append(int(annotation[11]))
        else:
            rate7.append(0)
        if int(cancer_rate) < int(annotation[12]):
            rate8.append(int(annotation[12]))
        else:
            rate8.append(0)
        if int(cancer_rate) < int(annotation[13]):
            rate9.append(int(annotation[13]))
        else:
            rate9.append(0)
    # for row in range(rows):
    #     for col in range(cols):
    #         for annotation in annotations:
    #             annotation = annotation.strip().split(' ')
    #             if int(row) == int(annotation[3]) and int(col) == int(annotation[2]):
    #                 background_choose = int(annotation[1])
    #                 color_alphas.append(background_choose)
    #                 cancer_type.append(int(annotation[4]))
    #                 if int(cancer_rate) < int(annotation[5]):
    #                     rate1.append(int(annotation[5]))
    #                 else:
    #                     rate1.append(0)
    #                 if int(cancer_rate) < int(annotation[6]):
    #                     rate2.append(int(annotation[6]))
    #                 else:
    #                     rate2.append(0)
    #                 if int(cancer_rate) < int(annotation[7]):
    #                     rate3.append(int(annotation[7]))
    #                 else:
    #                     rate3.append(0)
    #                 if int(cancer_rate) < int(annotation[8]):
    #                     rate4.append(int(annotation[8]))
    #                 else:
    #                     rate4.append(0)
    #                 if int(cancer_rate) < int(annotation[9]):
    #                     rate5.append(int(annotation[9]))
    #                 else:
    #                     rate5.append(0)
    #                 if int(cancer_rate) < int(annotation[10]):
    #                     rate6.append(int(annotation[10]))
    #                 else:
    #                     rate6.append(0)
    #                 if int(cancer_rate) < int(annotation[11]):
    #                     rate7.append(int(annotation[11]))
    #                 else:
    #                     rate7.append(0)
    #                 if int(cancer_rate) < int(annotation[12]):
    #                     rate8.append(int(annotation[12]))
    #                 else:
    #                     rate8.append(0)
    #                 if int(cancer_rate) < int(annotation[13]):
    #                     rate9.append(int(annotation[13]))
    #                 else:
    #                     rate9.append(0)

                # else:
                #     continue

        #         else:
        #             continue
        # continue
    return fix_area, color_alphas, cols, rows, cancer_type, \
           rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, \
           mouse_rects_dict, len(annotations)


# def build_rects_and_color_alphas_for_background(
#     grid_size_0_level, level_size_0, slide_name, slice_func=slice_rect2
# ):
#     svs_name = slide_name.split('/')[-1][:-4]
#     index_path = os.path.join('../dataprocess/data-index/', svs_name)  ##背景筛选信息位置 创建这个文件夹
#     background_index = index_path + '/' + 'svs_index.txt'
#     print(background_index)
#     with open(background_index, 'r') as f:
#         annotations = f.readlines()
#     color_alphas = []
#     for annotation in annotations:
#         annotation = annotation.strip().split(' ')
#         background_choose = int(annotation[1])
#         # print(background_choose)
#         color_alphas.append(background_choose)
#
#     rect_size = grid_size_0_level[0], grid_size_0_level[1]
#     print('buildbackground', rect_size)
#     rects, cols, rows = slice_func(level_size_0, rect_size, rect_size)  ##向内传参
#     # color_alphas = [(0, 255, 0, random.randint(0, 128)) for i in range(len(rects))]
#     # color_alphas = [(0, 255, 0, 0) for i in range(len(rects))]
#     # color_alphas = [0 for i in range(len(rects))]  ##无值闪退 有值为零 .rects为每一个小块坐标定为零保持显示完整性
#     # print('buildback',rects)   ##rect为画好的区块
#     # print(len(rects), color_alphas)
#     print('read')
#     return rects, color_alphas, cols, rows

# def pick_grid_change_index(
#     index_path, col, row
# ):
#
#     index_path = os.path.join('../dataprocess/data-index/', svs_name)  ##背景筛选信息位置 创建这个文件夹
#     background_index = index_path + '/' + 'svs_index.txt'
#     print(background_index)
#     with open(background_index, 'r') as f:
#         annotations = f.readlines()
#     color_alphas = []
#     for annotation in annotations:
#         annotation = annotation.strip().split(' ')
#         background_choose = int(annotation[1])
#         color_alphas.append(background_choose)
#
#     rect_size = grid_size_0_level[0], grid_size_0_level[1]
#     rects, cols, rows = slice_func(level_size_0, rect_size, rect_size)  ##向内传参
#
#     return rects, color_alphas, cols, rows


# def build_grid_level(level, grid_size_0_level, slide_helper: SlideHelper):
#     level_size = slide_helper.get_level_size(level)
#     level_downsample = slide_helper.get_downsample_for_level(level)
#     rect_size = grid_size_0_level[0] / level_downsample, grid_size_0_level[1] / level_downsample
#     rects = slice_rect(level_size, rect_size)

# colors = [(0, 255, 0, random.randint(0, 128)) for i in range(len(rects))]
# color_alphas=[random.randint(0, 128) for i in range(len(rects))]
# graphics_grid = GraphicsGrid(rects,
#                              colors,
#                              [0, 0, *level_size],
#                              color_alphas)
# return graphics_grid
def build_bin_level_from_rects(
        level, rects_0_level, color_alphas, unet_use, slide_helper: SlideHelper
):
    level_size = slide_helper.get_level_size(level)
    level_downsample = slide_helper.get_downsample_for_level(level)
    rects = [
        (
            int(rect_0_level[0] / level_downsample),
            int(rect_0_level[1] / level_downsample),
            int(rect_0_level[2] / level_downsample),
            int(rect_0_level[3] / level_downsample),

        )
        for rect_0_level in rects_0_level
    ]
    graphics_grid = GridBinfilterPaint(rects, color_alphas, unet_use, [0, 0, *level_size])
    print('filter-build-done', unet_use)
    return graphics_grid

def build_grid_level_from_rects(
        level, rects_0_level, intensities, cancer_type, distance, slide_helper: SlideHelper
):
    level_size = slide_helper.get_level_size(level)
    level_downsample = slide_helper.get_downsample_for_level(level)
    rects = [
        (
            int(rect_0_level[0] / level_downsample),
            int(rect_0_level[1] / level_downsample),
            int(rect_0_level[2] / level_downsample),
            int(rect_0_level[3] / level_downsample),

        )
        for rect_0_level in rects_0_level
    ]
    # graphics_grid = GridGraphicsItem(rects, intensities,
    #                                  [distance[0], distance[1], level_size[0]+distance[0], level_size[1]+distance[1]])
    graphics_grid = GridGraphicsItem(rects, intensities, cancer_type, [0, 0, *level_size])
    # print('levelbuilder', intensities)
    # print('build', graphics_grid)
    print('back-build-done')

    return graphics_grid


def build_type_select_from_rects(
        level, rects_0_level, intensities, cancer_type, select_type,
        rate1, rate2, rate3, rate4, rate5, rate6,
        rate7, rate8, rate9,
        clarity,
        slide_helper: SlideHelper
):
    level_size = slide_helper.get_level_size(level)
    level_downsample = slide_helper.get_downsample_for_level(level)
    rects = [
        (
            int(rect_0_level[0] / level_downsample),
            int(rect_0_level[1] / level_downsample),
            int(rect_0_level[2] / level_downsample),
            int(rect_0_level[3] / level_downsample),

        )
        for rect_0_level in rects_0_level
    ]
    # graphics_grid = GridGraphicsItem(rects, intensities,
    #                                  [distance[0], distance[1], level_size[0]+distance[0], level_size[1]+distance[1]])
    if len(select_type) == 1:
        graphics_grid = GridOnlyPaint(rects, intensities, cancer_type, [0, 0, *level_size], select_type,
                                      rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, clarity)
    else:    #多个类别多选
        type_rate = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9]
        # type = []
        # for x in select_type:
        #     type.append(type_rate[x-1])
        # print(len(type))
        alpha = []
        type = []
        for a in range(len(rate1)):
            c = []
            for x in select_type:
                c.append(type_rate[x - 1][a])
            max_index, max_number = max(enumerate(c), key=operator.itemgetter(1))
            alpha.append(max_number)
            type.append(select_type[max_index])  #
            continue
        graphics_grid = GridGraphicsItem(rects, alpha, type, [0, 0, *level_size], clarity)
    print('back-build-done')

    return graphics_grid
