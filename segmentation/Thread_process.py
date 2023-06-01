import sys
from PyQt5.QtCore import Qt, QThread,pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,QHBoxLayout


        # self.my_thread = MyThread()#实例化线程对象
        # self.my_thread.my_signal.connect(self.set_label_func)

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
from PyQt5.QtWidgets import *
import PyQt5




class MyThread(QThread):#线程类
    my_signal = pyqtSignal(str)  #自定义信号对象。参数str就代表这个信号可以传一个字符串
    def __init__(self):
        super(MyThread, self).__init__()
        self.is_on = True


    def run(self): #线程执行函数
        while self.is_on :
            self.my_signal.emit(str(self.count))  #释放自定义的信号
            self.cut_tiles()

    def cut_tiles(self, slidepath, tilesize, magnification, ratio):
        out_folder = '../dataprocess/slide_tiles/'
        self.get_tiles(slidepath, out_folder, tilesize, magnification, ratio, False)
        print('Done!')
        QMessageBox.information(None, "Items", '分割完成')
        return
        # 需要设置缓冲不再凸显运行时间弊端

    def check_background(self, tile, tile_size, ratio):

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

    def get_tiles(self, slide_name, out_folder, tile_size, magnification, ratio, color_normalization=False):
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
        partial_work = partial(self.pool_process, pool_param=pool_param)  # 偏函数打包
        pool.map(partial_work, row)
        pool.close()
        pool.join()
        t2 = time.time()
        print("并行执行时间：", int(t2 - t1))
        print('end')

    def pool_process(self, row, pool_param):
        slide_name, tile_size, ratio, out_path, index_path, dz_level, address = pool_param
        slide = openslide.open_slide(slide_name)
        slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)  ##全部二倍下采样补充完整，并完成分割计算
        pos_index = open(index_path + '/' + 'pos_index.txt', 'a')
        neg_index = open(index_path + '/' + 'neg_index.txt', 'a')
        svs_index = open(index_path + '/' + 'svs_index.txt', 'a')
        for col in range(address[0]):
            tile = slide_dz.get_tile(dz_level, (col, row))
            flag = self.check_background(tile, tile_size, ratio)  ##ratio 背景筛选参数
            if flag:
                p_mes = out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
                tile.save(out_path + '/' + str(col) + ' ' + str(row) + ' ' + '.jpg')
                pos_index.write(p_mes)
                s_mes = out_path + '/' + ' ' + '0' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'

            else:
                n_mes = out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
                tile.save(out_path + '/' + str(col) + ' ' + str(row) + ' ' + '.jpg')
                s_mes = out_path + '/' + ' ' + '255' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
                neg_index.write(n_mes)

            svs_index.write(s_mes)
        print('segmentation done')
        ##因多进程的使用无法反向传参只能在保存的list中读取进行信息获取