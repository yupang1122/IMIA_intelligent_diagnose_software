from wsi import slide
from wsi import filter
from wsi import tiles
import time
import os
import shutil
import easygui
from numpy import zeros

def background_filter_a(slide_path, bin_size, unet_use):
    back_folder = '../dataprocess/data-index/'
    svs_name = slide_path.split('/')[-1][:-4]
    back_path = os.path.join(back_folder, svs_name)  ##变量命名保存位置

    t1 = time.time()
    filter_path = os.path.join("../", "data")
    if not os.path.exists(filter_path):
        os.makedirs(filter_path)
    if not os.path.exists(back_path):
        os.makedirs(back_path)
    # 前背景分割
    shutil.rmtree(filter_path)  #文件夹清空
    slide.singleprocess_training_slides_to_images(slide_path)
    print('slidedone')
    bin_img = filter.singleprocess_apply_filters_to_images()
    print('filtersdone', bin_img.size)
    # tiles.singleprocess_filtered_images_to_tiles()
    # print('tilesdone')
    open(back_path + '/' + 'back_index.txt', 'w')
    ##背景过滤信息保存
    back_index = open(back_path + '/' + 'back_index.txt', 'a')
    for col in range(len(bin_img)):
        for row in range(len(bin_img[col])):
            back_mes = str(bin_img[col][row]) + ' '
            back_index.write(back_mes)
        back_index.write('0' + '\n')  #尾部加0 最后一行加0
    last_col = zeros((len(bin_img[0])), dtype = int)
    for i in range(len(last_col)):
        last_mes = str(last_col[i]) + ' '
        back_index.write(last_mes)
    print('backwritedone',len(last_col))
    back_index.write(last_mes)  # 尾部加0 最后一行加0
    t2 = time.time()
    print("去背景执行时间：", int(t2 - t1))
    print('backDone')
    easygui.msgbox("背景分割完成", '提示窗口')
    return bin_img