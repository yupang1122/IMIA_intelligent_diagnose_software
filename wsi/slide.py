import glob
import math
import openslide
# from openslide import OpenSlideError
import os
import PIL
from PIL import Image
import re
from wsi import util
from wsi.util import Time

BASE_DIR = os.path.join("../", "data")
# BASE_DIR = os.path.join(os.sep, "Volumes", "BigData", "TUPAC")
TRAIN_PREFIX = "TUPAC-TR-"
SRC_TRAIN_DIR = os.path.join(BASE_DIR, "training_slides")

# SRC_TRAIN_EXT = "svs"
DEST_TRAIN_SUFFIX = ""  # Example: "train-"
DEST_TRAIN_EXT = "png"
SCALE_FACTOR = 32
DEST_TRAIN_DIR = os.path.join(BASE_DIR, "training_" + DEST_TRAIN_EXT)
THUMBNAIL_SIZE = 300
THUMBNAIL_EXT = "jpg"

DEST_TRAIN_THUMBNAIL_DIR = os.path.join(BASE_DIR, "training_thumbnail_" + THUMBNAIL_EXT)

FILTER_SUFFIX = ""  # Example: "filter-"
FILTER_RESULT_TEXT = "filtered"
FILTER_DIR = os.path.join(BASE_DIR, "filter_" + DEST_TRAIN_EXT)
FILTER_THUMBNAIL_DIR = os.path.join(BASE_DIR, "filter_thumbnail_" + THUMBNAIL_EXT)
FILTER_PAGINATION_SIZE = 50
FILTER_PAGINATE = True
FILTER_HTML_DIR = BASE_DIR

TILE_SUMMARY_DIR = os.path.join(BASE_DIR, "tile_summary_" + DEST_TRAIN_EXT)
TILE_SUMMARY_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_" + DEST_TRAIN_EXT)
TILE_SUMMARY_SUFFIX = "tile_summary"
TILE_SUMMARY_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR, "tile_summary_on_original_thumbnail_" + THUMBNAIL_EXT)
TILE_SUMMARY_PAGINATION_SIZE = 50
TILE_SUMMARY_PAGINATE = True
TILE_SUMMARY_HTML_DIR = BASE_DIR

TILE_DATA_DIR = os.path.join(BASE_DIR, "tile_data")
TILE_DATA_SUFFIX = "tile_data"

TOP_TILES_SUFFIX = "top_tile_summary"
TOP_TILES_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_" + DEST_TRAIN_EXT)
TOP_TILES_THUMBNAIL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_thumbnail_" + THUMBNAIL_EXT)
TOP_TILES_ON_ORIGINAL_DIR = os.path.join(BASE_DIR, TOP_TILES_SUFFIX + "_on_original_" + DEST_TRAIN_EXT)
TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR = os.path.join(BASE_DIR,
                                                   TOP_TILES_SUFFIX + "_on_original_thumbnail_" + THUMBNAIL_EXT)

TILE_DIR = os.path.join(BASE_DIR, "tiles_" + DEST_TRAIN_EXT)
TILE_SUFFIX = "tile"

STATS_DIR = os.path.join(BASE_DIR, "svs_stats")


def open_slide(filename):
    slide = openslide.open_slide(filename)
    # try:
    #     slide = openslide.open_slide(filename)
    # except OpenSlideError:
    #     slide = None
    # except FileNotFoundError:
    #     slide = None
    return slide


def open_image(filename):

    image = Image.open(filename)
    return image


def open_image_np(filename):

    pil_img = open_image(filename)
    np_img = util.pil_to_np_rgb(pil_img)
    return np_img


def get_training_slide_path(slide_number):

    #不再给序号直接给定图片域名
    # padded_sl_num = str(slide_number).zfill(3)
    slide_filepath = os.path.join("../test-data", "tcg8.svs")
    print(slide_filepath)
    # slide_filepath = slide_path
    # slide_filepath = os.path.join(SRC_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "." + SRC_TRAIN_EXT)
    return slide_filepath
    # return

def get_tile_image_path(tile):

    t = tile
    padded_sl_num = str(t.slide_num).zfill(3)
    tile_path = os.path.join(TILE_DIR, padded_sl_num,
                             TRAIN_PREFIX + padded_sl_num + "-" + TILE_SUFFIX + "-r%d-c%d-x%d-y%d-w%d-h%d" % (
                                 t.r, t.c, t.o_c_s, t.o_r_s, t.o_c_e - t.o_c_s,
                                 t.o_r_e - t.o_r_s) + "." + DEST_TRAIN_EXT)
    return tile_path

def get_training_image_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):

    padded_sl_num = str(slide_number).zfill(3)
    if large_w is None and large_h is None and small_w is None and small_h is None:
        wildcard_path = os.path.join(DEST_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "*." + DEST_TRAIN_EXT)
        # print('wild', wildcard_path)
        img_path = glob.glob(wildcard_path)[0]
        # print('ip', img_path)
    else:
        img_path = os.path.join(DEST_TRAIN_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
            SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
            large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + DEST_TRAIN_EXT)
    return img_path


def get_training_thumbnail_path(slide_number, large_w=None, large_h=None, small_w=None, small_h=None):

    padded_sl_num = str(slide_number).zfill(3)
    if large_w is None and large_h is None and small_w is None and small_h is None:
        wilcard_path = os.path.join(DEST_TRAIN_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "*." + THUMBNAIL_EXT)
        img_path = glob.glob(wilcard_path)[0]
    else:
        img_path = os.path.join(DEST_TRAIN_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
            SCALE_FACTOR) + "x-" + DEST_TRAIN_SUFFIX + str(
            large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(small_h) + "." + THUMBNAIL_EXT)
    return img_path


def get_filter_image_path(slide_number, filter_number, filter_name_info):

    dir = FILTER_DIR
    if not os.path.exists(dir):
        os.makedirs(dir)
    img_path = os.path.join(dir, get_filter_image_filename(slide_number, filter_number, filter_name_info))
    return img_path


def get_filter_thumbnail_path(slide_number, filter_number, filter_name_info):

    dir = FILTER_THUMBNAIL_DIR
    if not os.path.exists(dir):
        os.makedirs(dir)
    img_path = os.path.join(dir,
                            get_filter_image_filename(slide_number, filter_number, filter_name_info, thumbnail=True))
    return img_path


def get_filter_image_filename(slide_number, filter_number, filter_name_info, thumbnail=False):

    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT
    padded_sl_num = str(slide_number).zfill(3)
    padded_fi_num = str(filter_number).zfill(3)
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + padded_fi_num + "-" + FILTER_SUFFIX + filter_name_info + "." + ext
    return img_filename


def get_tile_summary_image_path(slide_number):

    if not os.path.exists(TILE_SUMMARY_DIR):
        os.makedirs(TILE_SUMMARY_DIR)
    img_path = os.path.join(TILE_SUMMARY_DIR, get_tile_summary_image_filename(slide_number))
    return img_path


def get_tile_summary_thumbnail_path(slide_number):

    if not os.path.exists(TILE_SUMMARY_THUMBNAIL_DIR):
        os.makedirs(TILE_SUMMARY_THUMBNAIL_DIR)
    img_path = os.path.join(TILE_SUMMARY_THUMBNAIL_DIR, get_tile_summary_image_filename(slide_number, thumbnail=True))
    return img_path


def get_tile_summary_on_original_image_path(slide_number):

    if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_DIR):
        os.makedirs(TILE_SUMMARY_ON_ORIGINAL_DIR)
    img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_DIR, get_tile_summary_image_filename(slide_number))
    return img_path


def get_tile_summary_on_original_thumbnail_path(slide_number):

    if not os.path.exists(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR):
        os.makedirs(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR)
    img_path = os.path.join(TILE_SUMMARY_ON_ORIGINAL_THUMBNAIL_DIR,
                            get_tile_summary_image_filename(slide_number, thumbnail=True))
    return img_path


def get_top_tiles_on_original_image_path(slide_number):

    if not os.path.exists(TOP_TILES_ON_ORIGINAL_DIR):
        os.makedirs(TOP_TILES_ON_ORIGINAL_DIR)
    img_path = os.path.join(TOP_TILES_ON_ORIGINAL_DIR, get_top_tiles_image_filename(slide_number))
    return img_path


def get_top_tiles_on_original_thumbnail_path(slide_number):

    if not os.path.exists(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR):
        os.makedirs(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR)
    img_path = os.path.join(TOP_TILES_ON_ORIGINAL_THUMBNAIL_DIR,
                            get_top_tiles_image_filename(slide_number, thumbnail=True))
    return img_path


def get_tile_summary_image_filename(slide_number, thumbnail=False):

    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT
    padded_sl_num = str(slide_number).zfill(3)

    training_img_path = get_training_image_path(slide_number)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
        large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_SUMMARY_SUFFIX + "." + ext

    return img_filename


def get_top_tiles_image_filename(slide_number, thumbnail=False):

    if thumbnail:
        ext = THUMBNAIL_EXT
    else:
        ext = DEST_TRAIN_EXT
    padded_sl_num = str(slide_number).zfill(3)

    training_img_path = get_training_image_path(slide_number)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    img_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
        large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TOP_TILES_SUFFIX + "." + ext

    return img_filename


def get_top_tiles_image_path(slide_number):

    if not os.path.exists(TOP_TILES_DIR):
        os.makedirs(TOP_TILES_DIR)
    img_path = os.path.join(TOP_TILES_DIR, get_top_tiles_image_filename(slide_number))
    return img_path


def get_top_tiles_thumbnail_path(slide_number):

    if not os.path.exists(TOP_TILES_THUMBNAIL_DIR):
        os.makedirs(TOP_TILES_THUMBNAIL_DIR)
    img_path = os.path.join(TOP_TILES_THUMBNAIL_DIR, get_top_tiles_image_filename(slide_number, thumbnail=True))
    return img_path


def get_tile_data_filename(slide_number):

    padded_sl_num = str(slide_number).zfill(3)

    training_img_path = get_training_image_path(slide_number)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    data_filename = TRAIN_PREFIX + padded_sl_num + "-" + str(SCALE_FACTOR) + "x-" + str(large_w) + "x" + str(
        large_h) + "-" + str(small_w) + "x" + str(small_h) + "-" + TILE_DATA_SUFFIX + ".csv"

    return data_filename


def get_tile_data_path(slide_number):

    if not os.path.exists(TILE_DATA_DIR):
        os.makedirs(TILE_DATA_DIR)
    file_path = os.path.join(TILE_DATA_DIR, get_tile_data_filename(slide_number))
    return file_path


def get_filter_image_result(slide_number):

    padded_sl_num = str(slide_number).zfill(3)
    training_img_path = get_training_image_path(slide_number)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    img_path = os.path.join(FILTER_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
        small_h) + "-" + FILTER_RESULT_TEXT + "." + DEST_TRAIN_EXT)
    return img_path


def get_filter_thumbnail_result(slide_number):

    padded_sl_num = str(slide_number).zfill(3)
    training_img_path = get_training_image_path(slide_number)
    large_w, large_h, small_w, small_h = parse_dimensions_from_image_filename(training_img_path)
    img_path = os.path.join(FILTER_THUMBNAIL_DIR, TRAIN_PREFIX + padded_sl_num + "-" + str(
        SCALE_FACTOR) + "x-" + FILTER_SUFFIX + str(large_w) + "x" + str(large_h) + "-" + str(small_w) + "x" + str(
        small_h) + "-" + FILTER_RESULT_TEXT + "." + THUMBNAIL_EXT)
    return img_path


def parse_dimensions_from_image_filename(filename):

    m = re.match(".*-([\d]*)x([\d]*)-([\d]*)x([\d]*).*\..*", filename)
    large_w = int(m.group(1))
    large_h = int(m.group(2))
    small_w = int(m.group(3))
    small_h = int(m.group(4))
    return large_w, large_h, small_w, small_h


def small_to_large_mapping(small_pixel, large_dimensions):

    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
    return large_x, large_y


def training_slide_to_image(slide_number, slide_path):

    #按照序号传入训练数据
    #这一步只进行下采样到1/32的png，jpg保存处理
    img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_path)

    img_path = get_training_image_path(slide_number, large_w, large_h, new_w, new_h) #图片保存地址
    # print("Saving image to: " + img_path)
    if not os.path.exists(DEST_TRAIN_DIR):
        os.makedirs(DEST_TRAIN_DIR)
    img.save(img_path)

    thumbnail_path = get_training_thumbnail_path(slide_number, large_w, large_h, new_w, new_h)#小尺寸缩略图保存
    save_thumbnail(img, THUMBNAIL_SIZE, thumbnail_path)


def slide_to_scaled_pil_image(slide_number):

    slide_filepath = slide_number
    slide = open_slide(slide_filepath)

    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR) #下采样
    return img, large_w, large_h, new_w, new_h

def save_thumbnail(pil_img, size, path, display_path=False):

    max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
    img = pil_img.resize(max_size, PIL.Image.BILINEAR)
    dir = os.path.dirname(path)
    if dir != '' and not os.path.exists(dir):
        os.makedirs(dir)
    img.save(path)

def training_slide_range_to_images(slide_path):

    slide_num = 1
    training_slide_to_image(slide_num, slide_path)

    return

def singleprocess_training_slides_to_images(slide_filepath):

    t = Time()
    training_slide_range_to_images(slide_filepath)
    t.elapsed_display()



