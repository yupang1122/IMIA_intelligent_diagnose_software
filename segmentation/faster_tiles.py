import openslide
import numpy as np
import cv2 as cv
import time
from openslide import deepzoom
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import easygui
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import json
import os
from PIL import Image
from glob import glob
from torchvision import transforms
from histoslider.deepnet_process.deepnet import resnet18

def cut_tiles(slidepath, tilesize, magnification, ratio):
    out_folder = '../dataprocess/slide_tiles/'
    print('path', slidepath)
    get_tiles(slidepath, out_folder, tilesize, magnification, ratio, False)
    easygui.msgbox("分割完成", '提示窗口')
    print('Done!')

    return
    # 需要设置缓冲不再凸显运行时间弊端


def check_background(tile, tile_size, ratio):
    if tile.size != (tile_size, tile_size):
        return False
    gray = tile.convert('L')
    bw = gray.point(lambda x: 0 if x < 225 else 1, 'F')
    arr = np.array(np.asarray(bw))
    avgBkg = np.average(bw)
    if avgBkg < ratio:
        return True
    else:
        return False


def get_tiles(slide_name, out_folder, tile_size, magnification, ratio, color_normalization=False):
    slide = openslide.open_slide(slide_name)
    max_mag = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]  ##maxmag参数必须取整
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)  ##全部二倍下采样补充完整，并完成分割计算
    mag_num = np.log2(int(max_mag) / magnification) + 1  ##将分割的图像统一到同一尺度下进行算法处理
    if mag_num > 0:
        dz_level = int(slide_dz.level_count - mag_num)
    else:
        dz_level = int(slide_dz.level_count - 1)  ##找到下采样后对应层数    放大倍数magnification
    address = slide_dz.level_tiles[int(dz_level)]  # 找到对应层的分割行列数据 #为分割任务进行序号标记
    svs_name = slide_name.split('/')[-1][:-4]
    out_path = os.path.join(out_folder, svs_name)  ##变量命名保存位置
    index_path = os.path.join('../dataprocess/data-index/', svs_name)  ##背景筛选信息位置 创建这个文件夹
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    open(index_path + '/' + 'pos_index.txt', 'w')
    open(index_path + '/' + 'neg_index.txt', 'w')
    open(index_path + '/' + 'svs_index.txt', 'w')
    pool_param = [slide_name, tile_size, ratio, out_path, index_path, dz_level, address]
    print('start')
    pool_num = cpu_count() - 2
    t1 = time.time()
    pool = Pool(pool_num)
    # global pool_process#多线程添加线程池并行操作  cpu核心数
    # for row in range(0, address[1]):
    row = []
    for i in range(address[1]):
        row.append(i)
        continue
    partial_work = partial(pool_process, pool_param=pool_param)  # 偏函数打包
    pool.map(partial_work, row)
    pool.close()
    pool.join()
    t2 = time.time()
    print("并行执行时间：", int(t2 - t1))
    print('end')





def pool_process(row, pool_param):
    slide_name, tile_size, ratio, out_path, index_path, dz_level, address = pool_param
    slide = openslide.open_slide(slide_name)
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)  ##全部二倍下采样补充完整，并完成分割计算
    pos_index = open(index_path + '/' + 'pos_index.txt', 'a')
    neg_index = open(index_path + '/' + 'neg_index.txt', 'a')
    svs_index = open(index_path + '/' + 'svs_index.txt', 'a')
    for col in range(address[0]):
        tile = slide_dz.get_tile(dz_level, (col, row))
        flag = check_background(tile, tile_size, ratio)  ##ratio 背景筛选参数
        if flag and col != address[0] - 1 and row != address[1] - 1:
            p_mes = out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
            tile.save(out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg')
            data_transform = transforms.ToTensor()
            tile_PIL = Image.open(out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg')
            print(tile_PIL)
            trans_tile = data_transform(tile_PIL)
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                'E:/pathology/CRC9CLASS/CRC9CLASS_model_state/best9classtensor_adam0.9985_dropout_32.ckpt')
            new_dict = dict()
            for i in pretrained_dict:
                new_dict.update({i[7:]: pretrained_dict[i]})
            model = resnet18()
            num_ftrs = model.fc1.in_features
            model.fc1 = nn.Linear(num_ftrs, 9)
            model.load_state_dict(new_dict)
            model.to(device)
            dataset = trans_tile
            model.eval()
            # dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
            #加载数据的函数
            # for imgs in dataloader:
            imgs = dataset.to(device)
            print(imgs.size())
            with torch.no_grad():
                    outputs, features = model(imgs)
                    probs = F.softmax(outputs, dim=1)
                    labels = torch.argmax(probs, dim=1)
                    batch_log = zip(p_mes, probs.cpu().numpy().tolist(), features.cpu().numpy().tolist(),
                                    labels.cpu().numpy().tolist())
                    for log in batch_log:
                        cancer_rate = int(log[1][log[3]] * 255)
                        print(cancer_rate, log[3])
                        rate1 = int(log[1][0] * 255)
                        rate2 = int(log[1][1] * 255)
                        rate3 = int(log[1][2] * 255)
                        rate4 = int(log[1][3] * 255)
                        rate5 = int(log[1][4] * 255)
                        rate6 = int(log[1][5] * 255)
                        rate7 = int(log[1][6] * 255)
                        rate8 = int(log[1][7] * 255)
                        rate9 = int(log[1][8] * 255)
                        cancer_type = log[3] + 1
                        annotation = log[0].strip().split(' ')
                        col = annotation[1]
                        row = annotation[2]
                        s_mes = out_path + '/' + ' ' + str(cancer_rate) + ' ' + str(col) + ' ' + \
                                str(row) + ' ' + str(cancer_type) + \
                                ' ' + str(rate1) + \
                                ' ' + str(rate2) + \
                                ' ' + str(rate3) + \
                                ' ' + str(rate4) + \
                                ' ' + str(rate5) + \
                                ' ' + str(rate6) + \
                                ' ' + str(rate7) + \
                                ' ' + str(rate8) + \
                                ' ' + str(rate9) + \
                                ' ' + '.jpg' + '\n'
                        svs_index.write(s_mes)
            pos_index.write(p_mes)

        else:
            n_mes = out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
            s_mes = out_path + '/' + ' ' + '0' + ' ' + str(col) + ' ' + str(row) + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '0' + \
                    ' ' + '.jpg' + '\n'
            neg_index.write(n_mes)
            svs_index.write(s_mes)
        continue




