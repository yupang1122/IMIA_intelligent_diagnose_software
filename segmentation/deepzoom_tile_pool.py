import openslide
import numpy as np
import os
# import spams as sp
import cv2 as cv
from glob import glob
from PIL import Image
import time
from openslide import deepzoom
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import easygui
import tkinter
# from histoslider.deepnet_process.save_features import net_process

def cut_tiles(slidepath, tilesize, magnification, ratio):
    out_folder = '../dataprocess/slide_tiles/'
    get_tiles(slidepath, out_folder, tilesize, magnification, ratio, False)
    easygui.msgbox("分割完成", '提示窗口')
    print('Done!')
    # QMessageBox.information(None, "Items", '分割完成')
    # net_process()
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
            pos_index.write(p_mes)
            cancer_rate = int(np.random.uniform(0, 1) * 255)
            rate1 = int(np.random.uniform(0, 1) * 255)
            rate2 = int(np.random.uniform(0, 1) * 255)
            rate3 = int(np.random.uniform(0, 1) * 255)
            rate4 = int(np.random.uniform(0, 1) * 255)
            rate5 = int(np.random.uniform(0, 1) * 255)
            rate6 = int(np.random.uniform(0, 1) * 255)
            rate7 = int(np.random.uniform(0, 1) * 255)
            rate8 = int(np.random.uniform(0, 1) * 255)
            rate9 = int(np.random.uniform(0, 1) * 255)
            cancer_type = np.random.randint(1, 10)
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

        else:
            n_mes = out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
            # tile.save(out_path + '/' + str(col) + '_' + str(row) + '.jpg')
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

    # print('segmentation done')
    ##因多进程的使用无法反向传参只能在保存的list中读取进行信息获取


def toOD(Im):
    Im[Im == 0] = 255
    return -1 * np.log(Im / 255)


def toRGB(Im):
    return (255 * np.exp(-1 * Im)).astype(np.uint8)


def get_stain_matrix(Im, beta=0.15, alpha=1):
    Im = (Im[(Im > beta).any(axis=1), :])
    _, V = np.linalg.eigh(np.cov(Im, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(Im, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return HE / np.linalg.norm(HE, axis=1)[:, None]


def get_concentration(Im, stain_matrix, lamda=0.01, ):
    return sp.lasso(Im.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T


def get_hematoxylin(concentration, stain_matrix, shape):
    return (255 * np.exp(
        -1 * np.dot(concentration[:, 0].reshape(-1, 1), stain_matrix[0, :].reshape(-1, 3)).reshape(shape))).astype(
        np.uint8)


def get_eoxin(concentration, stain_matrix, shape):
    return (255 * np.exp(
        -1 * np.dot(concentration[:, 1].reshape(-1, 1), stain_matrix[1, :].reshape(-1, 3)).reshape(shape))).astype(
        np.uint8)


def get_target_max(target_image_name):
    target_image = cv.imread(target_image_name)
    target_image = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    target_od = toOD(target_image).reshape((-1, 3))
    target_stain_matrix = get_stain_matrix(target_od)
    target_concentration = get_concentration(target_od, stain_matrix=target_stain_matrix)
    target_max = np.percentile(target_concentration, 99, axis=0)
    return target_max, target_stain_matrix


def transform(simgname, target_max, target_stain_matrix, hema=1, eo=1):
    # source_image = cv.imread(simgname)
    # source_image = cv.cvtColor(source_image, cv.COLOR_BGR2RGB)
    source_image = np.array(simgname)
    shape = source_image.shape
    source_od = toOD(source_image).reshape((-1, 3))
    source_stain_matrix = get_stain_matrix(source_od)
    source_concentration = get_concentration(source_od, stain_matrix=source_stain_matrix)
    source_max = np.percentile(source_concentration, 99, axis=0)
    source_concentration *= (target_max / source_max)
    source_od = np.dot(source_concentration, target_stain_matrix)
    source_image = toRGB(source_od).reshape(shape)
    if hema:
        hematoxylin = get_hematoxylin(source_concentration, target_stain_matrix, shape)
    if eo:
        eoxin = get_eoxin(source_concentration, target_stain_matrix, shape)
    if hema and eo:
        return source_image, hematoxylin, eoxin
    if hema:
        return source_image, hematoxylin
    if eo:
        return source_image, eoxin
    return source_image


def main(slide_name):
    out_folder = '../dataprocess/slide_tiles/'
    print(slide_name)
    get_tiles(slide_name, out_folder, 224, 20, 0.5, False)


if __name__ == '__main__':
    # target_max, target_stain_matrix = get_target_max('/media/dell-workstation/data8T/Blackz/data_preprocess/image/50_13.jpg')
    slide_list = glob('../test-data/*.svs')
    pool = Pool(35)  ##多线程添加线程池并行操作  cpu核心数
    print(slide_list)
    pool.map(main, slide_list)
    pool.close()
    pool.join()
    print('poolDone!')
