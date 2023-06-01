import math
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from wsi import util
from wsi import filter
from wsi import slide
from wsi.util import Time

TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
NUM_TOP_TILES = 50

DISPLAY_TILE_SUMMARY_LABELS = False
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

FONT_PATH = "C:/Windows/Fonts/Arial/arial.ttf"
SUMMARY_TITLE_FONT_PATH = "C:/Windows/Fonts/Arial/arial.ttf"
SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330


def get_num_tiles(rows, cols, row_tile_size, col_tile_size):

  num_row_tiles = math.ceil(rows / row_tile_size)
  num_col_tiles = math.ceil(cols / col_tile_size)
  return num_row_tiles, num_col_tiles


def get_tile_indices(rows, cols, row_tile_size, col_tile_size):

  indices = list()
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  for r in range(0, num_row_tiles):
    start_r = r * row_tile_size
    end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
      start_c = c * col_tile_size
      end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
      indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
  return indices


def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):

  r = row_tile_size * num_row_tiles + title_area_height
  c = col_tile_size * num_col_tiles
  summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
  # add gray edges so that tile text does not get cut off
  summary_img.fill(120)
  # color title area white
  summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
  summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
  summary = util.np_to_pil(summary_img)
  return summary


def generate_tile_summaries(tile_sum, np_img, display=True, save_summary=False):

  z = 300  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  for t in tile_sum.tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

  summary_txt = summary_title(tile_sum) + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  if DISPLAY_TILE_SUMMARY_LABELS:
    count = 0
    for t in tile_sum.tiles:
      count += 1
      label = "R%d\nC%d" % (t.r, t.c)
      font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
      # drop shadow behind text
      draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)
      draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0), font=font)

      draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
      draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    save_tile_summary_image(summary, slide_num)
    save_tile_summary_on_original_image(summary_orig, slide_num)


def generate_top_tile_summaries(tile_sum, np_img, display=True, save_summary=False, show_top_stats=True,
                                label_all_tiles=LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY,
                                border_all_tiles=BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY):

  z = 300  # height of area at top of summary slide
  slide_num = tile_sum.slide_num
  rows = tile_sum.scaled_h
  cols = tile_sum.scaled_w
  row_tile_size = tile_sum.scaled_tile_h
  col_tile_size = tile_sum.scaled_tile_w
  num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
  summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw = ImageDraw.Draw(summary)

  original_img_path = slide.get_training_image_path(slide_num)
  np_orig = slide.open_image_np(original_img_path)
  summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
  draw_orig = ImageDraw.Draw(summary_orig)

  if border_all_tiles:
    for t in tile_sum.tiles:
      border_color = faded_tile_border_color(t.tissue_percentage)
      tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)
      tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color, border_size=1)

  tbs = TILE_BORDER_SIZE
  top_tiles = tile_sum.top_tiles()
  for t in top_tiles:
    border_color = tile_border_color(t.tissue_percentage)
    tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
    if border_all_tiles:
      tile_border(draw, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))
      tile_border(draw_orig, t.r_s + z + tbs, t.r_e + z - tbs, t.c_s + tbs, t.c_e - tbs, (0, 0, 0))

  summary_title = "Slide %03d Top Tile Summary:" % slide_num
  summary_txt = summary_title + "\n" + summary_stats(tile_sum)

  summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
  draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)
  draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR, font=summary_font)

  tiles_to_label = tile_sum.tiles if label_all_tiles else top_tiles
  h_offset = TILE_BORDER_SIZE + 2
  v_offset = TILE_BORDER_SIZE
  h_ds_offset = TILE_BORDER_SIZE + 3
  v_ds_offset = TILE_BORDER_SIZE + 1
  for t in tiles_to_label:
    label = "R%d\nC%d" % (t.r, t.c)
    font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
    # drop shadow behind text
    draw.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)
    draw_orig.text(((t.c_s + h_ds_offset), (t.r_s + v_ds_offset + z)), label, (0, 0, 0), font=font)

    draw.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)
    draw_orig.text(((t.c_s + h_offset), (t.r_s + v_offset + z)), label, SUMMARY_TILE_TEXT_COLOR, font=font)

  if show_top_stats:
    summary = add_tile_stats_to_top_tile_summary(summary, top_tiles, z)
    summary_orig = add_tile_stats_to_top_tile_summary(summary_orig, top_tiles, z)

  if display:
    summary.show()
    summary_orig.show()
  if save_summary:
    save_top_tiles_image(summary, slide_num)
    save_top_tiles_on_original_image(summary_orig, slide_num)


def add_tile_stats_to_top_tile_summary(pil_img, tiles, z):
  np_sum = util.pil_to_np_rgb(pil_img)
  sum_r, sum_c, sum_ch = np_sum.shape
  np_stats = np_tile_stat_img(tiles)
  st_r, st_c, _ = np_stats.shape
  combo_c = sum_c + st_c
  combo_r = max(sum_r, st_r + z)
  combo = np.zeros([combo_r, combo_c, sum_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:sum_r, 0:sum_c] = np_sum
  combo[z:st_r + z, sum_c:sum_c + st_c] = np_stats
  result = util.np_to_pil(combo)
  return result


def np_tile_stat_img(tiles):
  tt = sorted(tiles, key=lambda t: (t.r, t.c), reverse=False)
  tile_stats = "Tile Score Statistics:\n"
  count = 0
  for t in tt:
    if count > 0:
      tile_stats += "\n"
    count += 1
    tup = (t.r, t.c, t.rank, t.tissue_percentage, t.color_factor, t.s_and_v_factor, t.quantity_factor, t.score)
    tile_stats += "R%03d C%03d #%003d TP:%6.2f%% CF:%4.0f SVF:%4.2f QF:%4.2f S:%0.4f" % tup
  np_stats = np_text(tile_stats, font_path=SUMMARY_TITLE_FONT_PATH, font_size=14)
  return np_stats


def tile_border_color(tissue_percentage):
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = HIGH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = LOW_COLOR
  else:
    border_color = NONE_COLOR
  return border_color


def faded_tile_border_color(tissue_percentage):
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    border_color = FADED_THRESH_COLOR
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    border_color = FADED_MEDIUM_COLOR
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    border_color = FADED_LOW_COLOR
  else:
    border_color = FADED_NONE_COLOR
  return border_color


def summary_title(tile_summary):
  return "Slide %03d Tile Summary:" % tile_summary.slide_num


def summary_stats(tile_summary):
  return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
         "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
         "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
         "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
           tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
         "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
         " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
           tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
           tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
           TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
           tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
         " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)


def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
  for x in range(0, border_size):
    draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)


def save_tile_summary_image(pil_img, slide_num):
  t = Time()
  filepath = slide.get_tile_summary_image_path(slide_num)
  pil_img.save(filepath)
  # print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  # print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Thumb", str(t.elapsed()), thumbnail_filepath))


def save_top_tiles_image(pil_img, slide_num):
  t = Time()
  filepath = slide.get_top_tiles_image_path(slide_num)
  pil_img.save(filepath)
  # print("%-20s | Time: %-14s  Name: %s" % ("Save Top Tiles Image", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_top_tiles_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  # print("%-20s | Time: %-14s  Name: %s" % ("Save Top Tiles Thumb", str(t.elapsed()), thumbnail_filepath))


def save_tile_summary_on_original_image(pil_img, slide_num):
  t = Time()
  filepath = slide.get_tile_summary_on_original_image_path(slide_num)
  pil_img.save(filepath)
  # print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_tile_summary_on_original_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  # print(
  #   "%-20s | Time: %-14s  Name: %s" % ("Save Tile Sum Orig T", str(t.elapsed()), thumbnail_filepath))


def save_top_tiles_on_original_image(pil_img, slide_num):
  t = Time()
  filepath = slide.get_top_tiles_on_original_image_path(slide_num)
  pil_img.save(filepath)
  # print("%-20s | Time: %-14s  Name: %s" % ("Save Top Orig", str(t.elapsed()), filepath))

  t = Time()
  thumbnail_filepath = slide.get_top_tiles_on_original_thumbnail_path(slide_num)
  slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
  # print(
  #   "%-20s | Time: %-14s  Name: %s" % ("Save Top Orig Thumb", str(t.elapsed()), thumbnail_filepath))


def summary_and_tiles(slide_num, display=True, save_summary=False, save_data=True, save_top_tiles=True):
  img_path = slide.get_filter_image_result(slide_num)
  np_img = slide.open_image_np(img_path) ##数据格式转变

  tile_sum = score_tiles(slide_num, np_img) #tiles打分
  if save_data:
    save_tile_data(tile_sum)
  generate_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
  # if save_top_tiles:
  #   for tile in tile_sum.top_tiles():
  #     # print('save_tile',tile)
  #     tile.save_tile()
  return tile_sum


def save_tile_data(tile_summary):
  csv = summary_title(tile_summary) + "\n" + summary_stats(tile_summary)

  csv += "\n\n\nTile Num,Row,Column,Tissue %,Tissue Quantity,Col Start,Row Start,Col End,Row End,Col Size,Row Size," + \
         "Original Col Start,Original Row Start,Original Col End,Original Row End,Original Col Size,Original Row Size," + \
         "Color Factor,S and V Factor,Quantity Factor,Score\n"

  for t in tile_summary.tiles:
    line = "%d,%d,%d,%4.2f,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%4.0f,%4.2f,%4.2f,%0.4f\n" % (
      t.tile_num, t.r, t.c, t.tissue_percentage, t.tissue_quantity().name, t.c_s, t.r_s, t.c_e, t.r_e, t.c_e - t.c_s,
      t.r_e - t.r_s, t.o_c_s, t.o_r_s, t.o_c_e, t.o_r_e, t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s, t.color_factor,
      t.s_and_v_factor, t.quantity_factor, t.score)
    csv += line

  data_path = slide.get_tile_data_path(tile_summary.slide_num)
  csv_file = open(data_path, "w")
  csv_file.write(csv)
  csv_file.close()

  # print("%-20s | Time: %-14s  Name: %s" % ("Save Tile Data", str(time.elapsed()), data_path))


def tile_to_pil_tile(tile):
  t = tile
  # print(t.slide_num)
  slide_filepath = slide.get_training_slide_path(t.slide_num)

  s = slide.open_slide(slide_filepath)

  x, y = t.o_c_s, t.o_r_s
  w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
  tile_region = s.read_region((x, y), 0, (w, h))
  # RGBA to RGB
  pil_img = tile_region.convert("RGB")
  return pil_img


def tile_to_np_tile(tile):
  print('tile_to_np_tile')
  pil_img = tile_to_pil_tile(tile)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img


def save_display_tile(tile, save=True, display=False):
  print('save_display_tile')
  print(tile)
  tile_pil_img = tile_to_pil_tile(tile)

  if save:
    t = Time()
    img_path = slide.get_tile_image_path(tile)
    # print(img_path)  ##保存下来的区域小图
    dir = os.path.dirname(img_path)
    if not os.path.exists(dir):
      os.makedirs(dir)
    print(img_path)
    # tile_pil_img.save(img_path)
    # print("%-20s | Time: %-14s  Name: %s" % ("Save Tile", str(t.elapsed()), img_path))

  if display:
    tile_pil_img.show()


def score_tiles(slide_num, np_img=None, dimensions=None, small_tile_in_tile=False):
  if dimensions is None:
    img_path = slide.get_filter_image_result(slide_num)
    o_w, o_h, w, h = slide.parse_dimensions_from_image_filename(img_path)
  else:
    o_w, o_h, w, h = dimensions    #？？？废话？？
    #获取图像尺寸
  if np_img is None:
    np_img = slide.open_image_np(img_path)

  row_tile_size = round(ROW_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
  col_tile_size = round(COL_TILE_SIZE / slide.SCALE_FACTOR)  # use round?
  #scale缩放倍数   SCALE_FACTOR
  num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

  tile_sum = TileSummary(slide_num=slide_num,
                         orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=filter.tissue_percent(np_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0
  tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
  for t in tile_indices:
    count += 1  # tile_num
    r_s, r_e, c_s, c_e, r, c = t
    np_tile = np_img[r_s:r_e, c_s:c_e]
    t_p = filter.tissue_percent(np_tile)
    amount = tissue_quantity(t_p)
    if amount == TissueQuantity.HIGH:
      high += 1
    elif amount == TissueQuantity.MEDIUM:
      medium += 1
    elif amount == TissueQuantity.LOW:
      low += 1
    elif amount == TissueQuantity.NONE:
      none += 1
    o_c_s, o_r_s = slide.small_to_large_mapping((c_s, r_s), (o_w, o_h))
    o_c_e, o_r_e = slide.small_to_large_mapping((c_e, r_e), (o_w, o_h))

    # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
    if (o_c_e - o_c_s) > COL_TILE_SIZE:
      o_c_e -= 1
    if (o_r_e - o_r_s) > ROW_TILE_SIZE:
      o_r_e -= 1

    score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, slide_num, r, c)

    np_scaled_tile = np_tile if small_tile_in_tile else None
    tile = Tile(tile_sum, slide_num, np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
    tile_sum.tiles.append(tile)

  tile_sum.count = count
  tile_sum.high = high
  tile_sum.medium = medium
  tile_sum.low = low
  tile_sum.none = none

  tiles_by_score = tile_sum.tiles_by_score()
  rank = 0
  for t in tiles_by_score:
    rank += 1
    t.rank = rank

  return tile_sum


def score_tile(np_tile, tissue_percent, slide_num, row, col):
  color_factor = hsv_purple_pink_factor(np_tile)
  s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
  amount = tissue_quantity(tissue_percent)
  quantity_factor = tissue_quantity_factor(amount)
  combined_factor = color_factor * s_and_v_factor * quantity_factor
  score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
  # scale score to between 0 and 1
  score = 1.0 - (10.0 / (10.0 + score))
  return score, color_factor, s_and_v_factor, quantity_factor


def tissue_quantity_factor(amount):
  if amount == TissueQuantity.HIGH:
    quantity_factor = 1.0
  elif amount == TissueQuantity.MEDIUM:
    quantity_factor = 0.2
  elif amount == TissueQuantity.LOW:
    quantity_factor = 0.1
  else:
    quantity_factor = 0.0
  return quantity_factor


def tissue_quantity(tissue_percentage):
  if tissue_percentage >= TISSUE_HIGH_THRESH:
    return TissueQuantity.HIGH
  elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
    return TissueQuantity.MEDIUM
  elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
    return TissueQuantity.LOW
  else:
    return TissueQuantity.NONE


def image_range_to_tiles(start_ind, end_ind, display=False, save_summary=True, save_data=True, save_top_tiles=True):
  image_num_list = list()
  tile_summaries_dict = dict()
  for slide_num in range(start_ind, end_ind + 1):
    tile_summary = summary_and_tiles(slide_num, display, save_summary, save_data, save_top_tiles)
    image_num_list.append(slide_num)
    tile_summaries_dict[slide_num] = tile_summary
  return image_num_list, tile_summaries_dict


def singleprocess_filtered_images_to_tiles(display=False, save_summary=True, save_data=True, save_top_tiles=True,
                                           html=True, image_num_list=None):
    #直接调用会运行到这里设置只运行一张
    #文件夹目录下直接获取位置和文件总数
  num_training_slides = 1
  image_num_list, tile_summaries_dict = image_range_to_tiles(1, num_training_slides, display, save_summary, save_data,
                                                               save_top_tiles)

  if html:
    print('html')  #程序段只执行一次  对于单张图片的处理
    generate_tiled_html_result(image_num_list, tile_summaries_dict, save_data)  ##savedata 数据存储标志




def image_row(slide_num, tile_summary, data_link):
  orig_img = slide.get_training_image_path(slide_num)
  orig_thumb = slide.get_training_thumbnail_path(slide_num)
  filt_img = slide.get_filter_image_result(slide_num)
  filt_thumb = slide.get_filter_thumbnail_result(slide_num)
  sum_img = slide.get_tile_summary_image_path(slide_num)
  sum_thumb = slide.get_tile_summary_thumbnail_path(slide_num)
  osum_img = slide.get_tile_summary_on_original_image_path(slide_num)
  osum_thumb = slide.get_tile_summary_on_original_thumbnail_path(slide_num)
  top_img = slide.get_top_tiles_image_path(slide_num)
  top_thumb = slide.get_top_tiles_thumbnail_path(slide_num)
  otop_img = slide.get_top_tiles_on_original_image_path(slide_num)
  otop_thumb = slide.get_top_tiles_on_original_thumbnail_path(slide_num)
  html = "    <tr>\n" + \
         "      <td style=\"vertical-align: top\">\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Original<br/>\n" % (orig_img, slide_num) + \
         "          <img src=\"%s\" />\n" % (orig_thumb) + \
         "        </a>\n" + \
         "      </td>\n" + \
         "      <td style=\"vertical-align: top\">\n" + \
         "        <a target=\"_blank\" href=\"%s\">S%03d Filtered<br/>\n" % (filt_img, slide_num) + \
         "          <img src=\"%s\" />\n" % (filt_thumb) + \
         "        </a>\n" + \
         "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Tiles<br/>\n" % (sum_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (sum_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Tiles<br/>\n" % (osum_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (osum_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n"
  if data_link:
    html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary\n" % slide_num + \
            "        (<a target=\"_blank\" href=\"%s\">Data</a>)</div>\n" % slide.get_tile_data_path(slide_num)
  else:
    html += "        <div style=\"white-space: nowrap;\">S%03d Tile Summary</div>\n" % slide_num

  html += "        <div style=\"font-size: smaller; white-space: nowrap;\">\n" + \
          "          %s\n" % summary_stats(tile_summary).replace("\n", "<br/>\n          ") + \
          "        </div>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Top Tiles<br/>\n" % (top_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (top_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <a target=\"_blank\" href=\"%s\">S%03d Top Tiles<br/>\n" % (otop_img, slide_num) + \
          "          <img src=\"%s\" />\n" % (otop_thumb) + \
          "        </a>\n" + \
          "      </td>\n"

  top_tiles = tile_summary.top_tiles()
  num_tiles = len(top_tiles)
  score_num = 0
  for t in top_tiles:
    score_num += 1
    t.tile_num = score_num
  # sort top tiles by rows and columns to make them easier to locate on HTML page
  top_tiles = sorted(top_tiles, key=lambda t: (t.r, t.c), reverse=False)

  html += "      <td style=\"vertical-align: top\">\n" + \
          "        <div style=\"white-space: nowrap;\">S%03d Top %d Tile Scores</div>\n" % (slide_num, num_tiles) + \
          "        <div style=\"font-size: smaller; white-space: nowrap;\">\n"

  html += "        <table>\n"
  MAX_TILES_PER_ROW = 15
  num_cols = math.ceil(num_tiles / MAX_TILES_PER_ROW)
  num_rows = num_tiles if num_tiles < MAX_TILES_PER_ROW else MAX_TILES_PER_ROW
  for row in range(num_rows):
    html += "          <tr>\n"
    for col in range(num_cols):
      html += "            <td style=\"border: none;\">"
      tile_num = row + (col * num_rows) + 1
      if tile_num <= num_tiles:
        t = top_tiles[tile_num - 1]
        label = "R%03d C%03d %0.4f (#%02d)" % (t.r, t.c, t.score, t.tile_num)
        tile_img_path = slide.get_tile_image_path(t)
        html += "<a target=\"_blank\" href=\"%s\">%s</a>" % (tile_img_path, label)
      else:
        html += "&nbsp;"
      html += "</td>\n"
    html += "          </tr>\n"
  html += "        </table>\n"

  html += "        </div>\n"
  html += "      </td>\n"

  html += "    </tr>\n"
  return html


def generate_tiled_html_result(slide_nums, tile_summaries_dict, data_link):
  slide_nums = sorted(slide_nums)
  if not slide.TILE_SUMMARY_PAGINATE:
    print('generate_tiled_html_result_not')
    html = ""
    html += filter.html_header("Tiles")

    html += "  <table>\n"
    for slide_num in slide_nums:
      html += image_row(slide_num, data_link)
    html += "  </table>\n"

    html += filter.html_footer()
    text_file = open(os.path.join(slide.TILE_SUMMARY_HTML_DIR, "tiles.html"), "w")
    text_file.write(html)
    text_file.close()
  else:
    total_len = len(slide_nums)
    page_size = slide.TILE_SUMMARY_PAGINATION_SIZE
    num_pages = math.ceil(total_len / page_size)
    for page_num in range(1, num_pages + 1):
      start_index = (page_num - 1) * page_size
      end_index = (page_num * page_size) if (page_num < num_pages) else total_len
      page_slide_nums = slide_nums[start_index:end_index]

      html = ""
      html += filter.html_header("Tiles, Page %d" % page_num)

      html += "  <div style=\"font-size: 20px\">"
      if page_num > 1:
        if page_num == 2:
          html += "<a href=\"tiles.html\">&lt;</a> "
        else:
          html += "<a href=\"tiles-%d.html\">&lt;</a> " % (page_num - 1)
      html += "Page %d" % page_num
      if page_num < num_pages:
        html += " <a href=\"tiles-%d.html\">&gt;</a> " % (page_num + 1)
      html += "</div>\n"

      html += "  <table>\n"
      for slide_num in page_slide_nums:
        tile_summary = tile_summaries_dict[slide_num]
        html += image_row(slide_num, tile_summary, data_link)
      html += "  </table>\n"

      html += filter.html_footer()
      if page_num == 1:
        text_file = open(os.path.join(slide.TILE_SUMMARY_HTML_DIR, "tiles.html"), "w")
      else:
        text_file = open(os.path.join(slide.TILE_SUMMARY_HTML_DIR, "tiles-%d.html" % page_num), "w")
      text_file.write(html)
      text_file.close()


def np_hsv_hue_histogram(h):
  print(h, 'use')
  # figure = plt.figure()
  # canvas = figure.canvas
  # _, _, patches = plt.hist(h, bins=360)
  # plt.title("HSV Hue Histogram, mean=%3.1f, std=%3.1f" % (np.mean(h), np.std(h)))
  #
  # bin_num = 0
  # for patch in patches:
  #   rgb_color = colorsys.hsv_to_rgb(bin_num / 360.0, 1, 1)
  #   patch.set_facecolor(rgb_color)
  #   bin_num += 1
  #
  # canvas.draw()
  # w, h = canvas.get_width_height()
  # np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
  # plt.close(figure)
  # util.np_info(np_hist)
  # return np_hist
  return

def np_histogram(data, title, bins="auto"):
  print(data, title, 'use')
  # figure = plt.figure()
  # canvas = figure.canvas
  # plt.hist(data, bins=bins)
  # plt.title(title)
  #
  # canvas.draw()
  # w, h = canvas.get_width_height()
  # np_hist = np.fromstring(canvas.get_renderer().tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
  # plt.close(figure)
  # util.np_info(np_hist)
  # return np_hist
  return

def np_hsv_saturation_histogram(s):
  title = "HSV Saturation Histogram, mean=%.2f, std=%.2f" % (np.mean(s), np.std(s))
  return np_histogram(s, title)


def np_hsv_value_histogram(v):
  title = "HSV Value Histogram, mean=%.2f, std=%.2f" % (np.mean(v), np.std(v))
  return np_histogram(v, title)


def np_rgb_channel_histogram(rgb, ch_num, ch_name):
  ch = rgb[:, :, ch_num]
  ch = ch.flatten()
  title = "RGB %s Histogram, mean=%.2f, std=%.2f" % (ch_name, np.mean(ch), np.std(ch))
  return np_histogram(ch, title, bins=256)


def np_rgb_r_histogram(rgb):
  hist = np_rgb_channel_histogram(rgb, 0, "R")
  return hist


def np_rgb_g_histogram(rgb):
  hist = np_rgb_channel_histogram(rgb, 1, "G")
  return hist


def np_rgb_b_histogram(rgb):
  hist = np_rgb_channel_histogram(rgb, 2, "B")
  return hist


def display_image(np_rgb, text=None, scale_up=False):
  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)
  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i

  pil_img = util.np_to_pil(np_rgb)
  pil_img.show()


def display_image_with_hsv_histograms(np_rgb, text=None, scale_up=False):
  hsv = filter.filter_rgb_to_hsv(np_rgb)
  np_h = np_hsv_hue_histogram(filter.filter_hsv_to_h(hsv))
  np_s = np_hsv_saturation_histogram(filter.filter_hsv_to_s(hsv))
  np_v = np_hsv_value_histogram(filter.filter_hsv_to_v(hsv))
  h_r, h_c, _ = np_h.shape
  s_r, s_c, _ = np_s.shape
  v_r, v_c, _ = np_v.shape

  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i  # for simplicity assign title+image to image
    img_r, img_c, img_ch = np_rgb.shape

  hists_c = max(h_c, s_c, v_c)
  hists_r = h_r + s_r + v_r
  hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

  hists[0:h_r, 0:h_c] = np_h
  hists[h_r:h_r + s_r, 0:s_c] = np_s
  hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

  r = max(img_r, hists_r)
  c = img_c + hists_c
  combo = np.zeros([r, c, img_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:img_r, 0:img_c] = np_rgb
  combo[0:hists_r, img_c:c] = hists
  pil_combo = util.np_to_pil(combo)
  pil_combo.show()


def display_image_with_rgb_histograms(np_rgb, text=None, scale_up=False):
  np_r = np_rgb_r_histogram(np_rgb)
  np_g = np_rgb_g_histogram(np_rgb)
  np_b = np_rgb_b_histogram(np_rgb)
  r_r, r_c, _ = np_r.shape
  g_r, g_c, _ = np_g.shape
  b_r, b_c, _ = np_b.shape

  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i  # for simplicity assign title+image to image
    img_r, img_c, img_ch = np_rgb.shape

  hists_c = max(r_c, g_c, b_c)
  hists_r = r_r + g_r + b_r
  hists = np.zeros([hists_r, hists_c, img_ch], dtype=np.uint8)

  hists[0:r_r, 0:r_c] = np_r
  hists[r_r:r_r + g_r, 0:g_c] = np_g
  hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

  r = max(img_r, hists_r)
  c = img_c + hists_c
  combo = np.zeros([r, c, img_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:img_r, 0:img_c] = np_rgb
  combo[0:hists_r, img_c:c] = hists
  pil_combo = util.np_to_pil(combo)
  pil_combo.show()


def pil_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
             font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):

  font = ImageFont.truetype(font_path, font_size)
  x, y = ImageDraw.Draw(Image.new("RGB", (1, 1), background)).textsize(text, font)
  image = Image.new("RGB", (x + 2 * w_border, y + 2 * h_border), background)
  draw = ImageDraw.Draw(image)
  draw.text((w_border, h_border), text, text_color, font=font)
  return image


def np_text(text, w_border=TILE_TEXT_W_BORDER, h_border=TILE_TEXT_H_BORDER, font_path=FONT_PATH,
            font_size=TILE_TEXT_SIZE, text_color=TILE_TEXT_COLOR, background=TILE_TEXT_BACKGROUND_COLOR):
  pil_img = pil_text(text, w_border, h_border, font_path, font_size,
                     text_color, background)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img


def display_tile(tile, rgb_histograms=True, hsv_histograms=True):
  text = "S%03d R%03d C%03d\n" % (tile.slide_num, tile.r, tile.c)
  text += "Score:%4.2f Tissue:%5.2f%% CF:%2.0f SVF:%4.2f QF:%4.2f\n" % (
    tile.score, tile.tissue_percentage, tile.color_factor, tile.s_and_v_factor, tile.quantity_factor)
  text += "Rank #%d of %d" % (tile.rank, tile.tile_summary.num_tiles())

  np_scaled_tile = tile.get_np_scaled_tile()
  if np_scaled_tile is not None:
    small_text = text + "\n \nSmall Tile (%d x %d)" % (np_scaled_tile.shape[1], np_scaled_tile.shape[0])
    if rgb_histograms and hsv_histograms:
      display_image_with_rgb_and_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
    elif rgb_histograms:
      display_image_with_rgb_histograms(np_scaled_tile, small_text, scale_up=True)
    elif hsv_histograms:
      display_image_with_hsv_histograms(np_scaled_tile, small_text, scale_up=True)
    else:
      display_image(np_scaled_tile, small_text, scale_up=True)

  np_tile = tile.get_np_tile()
  text += " based on small tile\n \nLarge Tile (%d x %d)" % (np_tile.shape[1], np_tile.shape[0])
  if rgb_histograms and hsv_histograms:
    display_image_with_rgb_and_hsv_histograms(np_tile, text)
  elif rgb_histograms:
    display_image_with_rgb_histograms(np_tile, text)
  elif hsv_histograms:
    display_image_with_hsv_histograms(np_tile, text)
  else:
    display_image(np_tile, text)


def display_image_with_rgb_and_hsv_histograms(np_rgb, text=None, scale_up=False):
  hsv = filter.filter_rgb_to_hsv(np_rgb)
  np_r = np_rgb_r_histogram(np_rgb)
  np_g = np_rgb_g_histogram(np_rgb)
  np_b = np_rgb_b_histogram(np_rgb)
  np_h = np_hsv_hue_histogram(filter.filter_hsv_to_h(hsv))
  np_s = np_hsv_saturation_histogram(filter.filter_hsv_to_s(hsv))
  np_v = np_hsv_value_histogram(filter.filter_hsv_to_v(hsv))

  r_r, r_c, _ = np_r.shape
  g_r, g_c, _ = np_g.shape
  b_r, b_c, _ = np_b.shape
  h_r, h_c, _ = np_h.shape
  s_r, s_c, _ = np_s.shape
  v_r, v_c, _ = np_v.shape

  if scale_up:
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=1)
    np_rgb = np.repeat(np_rgb, slide.SCALE_FACTOR, axis=0)

  img_r, img_c, img_ch = np_rgb.shape
  if text is not None:
    np_t = np_text(text)
    t_r, t_c, _ = np_t.shape
    t_i_c = max(t_c, img_c)
    t_i_r = t_r + img_r
    t_i = np.zeros([t_i_r, t_i_c, img_ch], dtype=np.uint8)
    t_i.fill(255)
    t_i[0:t_r, 0:t_c] = np_t
    t_i[t_r:t_r + img_r, 0:img_c] = np_rgb
    np_rgb = t_i  # for simplicity assign title+image to image
    img_r, img_c, img_ch = np_rgb.shape

  rgb_hists_c = max(r_c, g_c, b_c)
  rgb_hists_r = r_r + g_r + b_r
  rgb_hists = np.zeros([rgb_hists_r, rgb_hists_c, img_ch], dtype=np.uint8)
  rgb_hists[0:r_r, 0:r_c] = np_r
  rgb_hists[r_r:r_r + g_r, 0:g_c] = np_g
  rgb_hists[r_r + g_r:r_r + g_r + b_r, 0:b_c] = np_b

  hsv_hists_c = max(h_c, s_c, v_c)
  hsv_hists_r = h_r + s_r + v_r
  hsv_hists = np.zeros([hsv_hists_r, hsv_hists_c, img_ch], dtype=np.uint8)
  hsv_hists[0:h_r, 0:h_c] = np_h
  hsv_hists[h_r:h_r + s_r, 0:s_c] = np_s
  hsv_hists[h_r + s_r:h_r + s_r + v_r, 0:v_c] = np_v

  r = max(img_r, rgb_hists_r, hsv_hists_r)
  c = img_c + rgb_hists_c + hsv_hists_c
  combo = np.zeros([r, c, img_ch], dtype=np.uint8)
  combo.fill(255)
  combo[0:img_r, 0:img_c] = np_rgb
  combo[0:rgb_hists_r, img_c:img_c + rgb_hists_c] = rgb_hists
  combo[0:hsv_hists_r, img_c + rgb_hists_c:c] = hsv_hists
  pil_combo = util.np_to_pil(combo)
  pil_combo.show()


def rgb_to_hues(rgb):
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  h = filter.filter_hsv_to_h(hsv, display_np_info=False)
  return h


def hsv_saturation_and_value_factor(rgb):
  hsv = filter.filter_rgb_to_hsv(rgb, display_np_info=False)
  s = filter.filter_hsv_to_s(hsv)
  v = filter.filter_hsv_to_v(hsv)
  s_std = np.std(s)
  v_std = np.std(v)
  if s_std < 0.05 and v_std < 0.05:
    factor = 0.4
  elif s_std < 0.05:
    factor = 0.7
  elif v_std < 0.05:
    factor = 0.7
  else:
    factor = 1

  factor = factor ** 2
  return factor


def hsv_purple_deviation(hsv_hues):
  purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
  return purple_deviation


def hsv_pink_deviation(hsv_hues):
  pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
  return pink_deviation


def hsv_purple_pink_factor(rgb):
  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 260]  # exclude hues under 260
  hues = hues[hues <= 340]  # exclude hues over 340
  if len(hues) == 0:
    return 0  # if no hues between 260 and 340, then not purple or pink
  pu_dev = hsv_purple_deviation(hues)
  pi_dev = hsv_pink_deviation(hues)
  avg_factor = (340 - np.average(hues)) ** 2

  if pu_dev == 0:  # avoid divide by zero if tile has no tissue
    return 0

  factor = pi_dev / pu_dev * avg_factor
  return factor


def hsv_purple_vs_pink_average_factor(rgb, tissue_percentage):
  factor = 1
  # only applies to slides with a high quantity of tissue
  if tissue_percentage < TISSUE_HIGH_THRESH:
    return factor

  hues = rgb_to_hues(rgb)
  hues = hues[hues >= 200]  # Remove hues under 200
  if len(hues) == 0:
    return factor
  avg = np.average(hues)
  # pil_hue_histogram(hues).show()

  pu = HSV_PURPLE - avg
  pi = HSV_PINK - avg
  pupi = pu + pi
  # print("Av: %4d, Pu: %4d, Pi: %4d, PuPi: %4d" % (avg, pu, pi, pupi))
  # Av:  250, Pu:   20, Pi:   80, PuPi:  100
  # Av:  260, Pu:   10, Pi:   70, PuPi:   80
  # Av:  270, Pu:    0, Pi:   60, PuPi:   60 ** PURPLE
  # Av:  280, Pu:  -10, Pi:   50, PuPi:   40
  # Av:  290, Pu:  -20, Pi:   40, PuPi:   20
  # Av:  300, Pu:  -30, Pi:   30, PuPi:    0
  # Av:  310, Pu:  -40, Pi:   20, PuPi:  -20
  # Av:  320, Pu:  -50, Pi:   10, PuPi:  -40
  # Av:  330, Pu:  -60, Pi:    0, PuPi:  -60 ** PINK
  # Av:  340, Pu:  -70, Pi:  -10, PuPi:  -80
  # Av:  350, Pu:  -80, Pi:  -20, PuPi: -100

  if pupi > 30:
    factor *= 1.2
  if pupi < -30:
    factor *= .8
  if pupi > 0:
    factor *= 1.2
  if pupi > 50:
    factor *= 1.2
  if pupi < -60:
    factor *= .8

  return factor


class TileSummary:
  slide_num = None
  orig_w = None
  orig_h = None
  orig_tile_w = None
  orig_tile_h = None
  scale_factor = slide.SCALE_FACTOR
  scaled_w = None
  scaled_h = None
  scaled_tile_w = None
  scaled_tile_h = None
  mask_percentage = None
  num_row_tiles = None
  num_col_tiles = None

  count = 0
  high = 0
  medium = 0
  low = 0
  none = 0

  def __init__(self, slide_num, orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles):
    self.slide_num = slide_num
    self.orig_w = orig_w
    self.orig_h = orig_h
    self.orig_tile_w = orig_tile_w
    self.orig_tile_h = orig_tile_h
    self.scaled_w = scaled_w
    self.scaled_h = scaled_h
    self.scaled_tile_w = scaled_tile_w
    self.scaled_tile_h = scaled_tile_h
    self.tissue_percentage = tissue_percentage
    self.num_col_tiles = num_col_tiles
    self.num_row_tiles = num_row_tiles
    self.tiles = []

  def __str__(self):
    return summary_title(self) + "\n" + summary_stats(self)

  def mask_percentage(self):
    return 100 - self.tissue_percentage

  def num_tiles(self):
    return self.num_row_tiles * self.num_col_tiles

  def tiles_by_tissue_percentage(self):
    sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
    return sorted_list

  def tiles_by_score(self):
    sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
    return sorted_list

  def top_tiles(self):
    sorted_tiles = self.tiles_by_score()
    top_tiles = sorted_tiles[:NUM_TOP_TILES]
    return top_tiles


class Tile:
  def __init__(self, tile_summary, slide_num, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
    self.tile_summary = tile_summary
    self.slide_num = slide_num
    self.np_scaled_tile = np_scaled_tile
    self.tile_num = tile_num
    self.r = r
    self.c = c
    self.r_s = r_s
    self.r_e = r_e
    self.c_s = c_s
    self.c_e = c_e
    self.o_r_s = o_r_s
    self.o_r_e = o_r_e
    self.o_c_s = o_c_s
    self.o_c_e = o_c_e
    self.tissue_percentage = t_p
    self.color_factor = color_factor
    self.s_and_v_factor = s_and_v_factor
    self.quantity_factor = quantity_factor
    self.score = score

  def __str__(self):
    return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

  def __repr__(self):
    return "\n" + self.__str__()

  def mask_percentage(self):
    return 100 - self.tissue_percentage

  def tissue_quantity(self):
    return tissue_quantity(self.tissue_percentage)

  def get_pil_tile(self):
    print('get_pil_tile')
    return tile_to_pil_tile(self)

  def get_np_tile(self):
    return tile_to_np_tile(self)

  def save_tile(self):
    # save_display_tile(self, save=True, display=False)
    return
  def display_tile(self):
    # save_display_tile(self, save=False, display=True)
    return
  def display_with_histograms(self):
    display_tile(self, rgb_histograms=True, hsv_histograms=True)

  def get_np_scaled_tile(self):
    return self.np_scaled_tile

  def get_pil_scaled_tile(self):
    return util.np_to_pil(self.np_scaled_tile)

class TissueQuantity(Enum):
  NONE = 0
  LOW = 1
  MEDIUM = 2
  HIGH = 3