import openslide
import numpy as np
import cv2 as cv
import time
import spams as sp
from openslide import deepzoom
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import easygui
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
# from torch import cuda, load, argmax, no_grad
# from torch.nn import Linear
# import torch.nn.functional as F
# import torch.utils.data as data
# import json
import os
from PIL import Image
from glob import glob

# from wsi import slide
# from wsi import filter
# from wsi import tiles

import shutil

from torchvision import transforms
from models.resnet_process_224.deepnet import resnet18 as resnet224
from models.tiny_resnet_56.Tinyresnet import resnet18 as resnet56
from models.tiny_resnet_112.Tinyresnet import resnet18 as resnet112
from models.tiny_resnet_56_2.Tinyresnet56 import resnet18 as resnet56_2
from torchvision import models

def cut_tiles(slide_path, tile_size, magnification, ratio):
    out_folder = '../dataprocess/slide_tiles/'
    # result_folder = '../dataprocess/deep_result/'
    index_folder = '../dataprocess/data-index/'
    svs_name = slide_path.split('/')[-1][:-4]
    out_path = os.path.join(out_folder, svs_name, str(tile_size))  ##变量命名保存位置
    index_path = os.path.join(index_folder, svs_name, str(tile_size))
    t1 = time.time()

    # # 前背景分割
    # shutil.rmtree("../data")  #文件夹清空
    # slide.singleprocess_training_slides_to_images(slide_path)
    # print('slidedone')
    # filter.singleprocess_apply_filters_to_images()
    # print('filtersdone')
    # tiles.singleprocess_filtered_images_to_tiles()
    # print('tilesdone')

    if not os.path.exists(out_path):
       get_tiles(slide_path, out_path, index_path, tile_size, magnification, ratio, False)
       print('segDone!',tile_size)

    # print(svs_name)
    # QMessageBox.information(None, "Items", '分割完成')
    # if not os.path.exists(index_path):
    net_process(out_path, index_path, tile_size)

    t2 = time.time()
    print("并行执行时间：", int(t2 - t1))
    print('netprocessDone')
    easygui.msgbox("分割完成", '提示窗口')
    return
    # 需要设置缓冲不再凸显运行时间弊端
    # 变量命名精简过了

def check_background(tile, tile_size, ratio, back_threshold):
    if tile.size != (tile_size, tile_size):
        return False
    gray = tile.convert('L')
    bw = gray.point(lambda x: 0 if x < int(back_threshold) else 1, 'F')
    arr = np.array(np.asarray(bw))
    avgBkg = np.average(bw)
    if avgBkg < ratio:
        return True
    else:
        return False

def toOD(Im):
	Im[Im == 0] = 255
	return -1*np.log(Im/255)

def toRGB(Im):
	return (255*np.exp(-1*Im)).astype(np.uint8)

def get_stain_matrix(Im, beta = 0.15, alpha = 1):
    Im = (Im[(Im > beta).any(axis = 1), :])
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
    return HE/np.linalg.norm(HE, axis=1)[:, None]

def get_concentration(Im, stain_matrix, lamda = 0.01,):
	return sp.lasso(Im.T, D=stain_matrix.T, mode=2, lambda1=lamda, pos=True).toarray().T

def get_hematoxylin(concentration, stain_matrix, shape):
	return (255 * np.exp(-1 * np.dot(concentration[:, 0].reshape(-1,1), stain_matrix[0,:].reshape(-1,3)).reshape(shape))).astype(np.uint8)

def get_eoxin(concentration, stain_matrix, shape):
	return (255 * np.exp(-1 * np.dot(concentration[:, 1].reshape(-1,1), stain_matrix[1,:].reshape(-1,3)).reshape(shape))).astype(np.uint8)

def get_target_max(target_image_name):
    target_image = cv.imread(target_image_name)
    target_image = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    target_od = toOD(target_image).reshape((-1,3))
    target_stain_matrix = get_stain_matrix(target_od)
    target_concentration = get_concentration(target_od, stain_matrix = target_stain_matrix)
    target_max = np.percentile(target_concentration, 99, axis = 0)
    return target_max, target_stain_matrix

def transform(simgname, target_max, target_stain_matrix, hema = 1, eo = 1):
    #source_image = cv.imread(simgname)
    #source_image = cv.cvtColor(source_image, cv.COLOR_BGR2RGB)
    source_image = np.array(simgname)
    shape = source_image.shape
    source_od = toOD(source_image).reshape((-1,3))
    source_stain_matrix = get_stain_matrix(source_od)
    source_concentration = get_concentration(source_od, stain_matrix = source_stain_matrix)
    source_max = np.percentile(source_concentration, 99, axis = 0)
    source_concentration *= (target_max/source_max)
    source_od = np.dot(source_concentration, target_stain_matrix)
    source_image = toRGB(source_od).reshape(shape)
    if hema:
        hematoxylin = get_hematoxylin(source_concentration, target_stain_matrix, shape)
    if eo:
        eoxin = get_eoxin(source_concentration, target_stain_matrix, shape)
    if hema and eo:
        return source_image,hematoxylin,eoxin
    if hema:
        return source_image,hematoxylin
    if eo:
        return source_image,eoxin
    return source_image


def get_tiles(slide_path, out_path, index_path, tile_size, magnification, ratio, color_normalization=False):
    slide = openslide.open_slide(slide_path)
    max_mag = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]  ##maxmag参数必须取整
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)  ##全部二倍下采样补充完整，并完成分割计算
    #这里是一个实例化无法作为参数传参

    mag_num = np.log2(int(max_mag) / magnification) + 1  ##将分割的图像统一到同一尺度下进行算法处理
    if mag_num > 0:
        dz_level = int(slide_dz.level_count - mag_num)
    else:
        dz_level = int(slide_dz.level_count - 1)  ##找到下采样后对应层数    放大倍数magnification
    address = slide_dz.level_tiles[int(dz_level)]  # 找到对应层的分割行列数据 #为分割任务进行序号标记

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    open(index_path + '/' + 'pos_index.txt', 'w')
    open(index_path + '/' + 'neg_index.txt', 'w')  #创建list


    tn = slide.get_thumbnail(slide.level_dimensions[-1])
    tn_array = np.array(tn)
    gray = cv.cvtColor(tn_array, cv.COLOR_RGB2GRAY)
    back_threshold, th = cv.threshold(gray, 0, 1, cv.THRESH_OTSU)
    print(back_threshold, 'process')

    normalization_target_path = os.path.abspath('colornorm1.jpg')

    target_max, target_stain_matrix = get_target_max(normalization_target_path)
    pool_param = [slide_path, tile_size, ratio, out_path, index_path, dz_level, address, target_max, target_stain_matrix, back_threshold]
    # print('start', address, max_mag, dz_level)
    pool_num = cpu_count() - 2
    pool = Pool(pool_num)
    # global pool_process#多线程添加线程池并行操作  cpu核心数
    # for row in range(0, address[1]):
    row = []
    for i in range(address[1]):
        row.append(i)
        continue
    #按数组元素传进程池
    partial_work = partial(pool_process, pool_param=pool_param)  # 偏函数打包
    pool.map(partial_work, row)
    pool.close()
    pool.join()

    # print('end')

def pool_process(row, pool_param):
    slide_name, tile_size, ratio, out_path, index_path, dz_level, address, target_max, target_stain_matrix, back_threshold = pool_param

    slide = openslide.open_slide(slide_name)
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)  ##全部二倍下采样补充完整，并完成分割计算
    pos_index = open(index_path + '/' + 'pos_index.txt', 'a')
    neg_index = open(index_path + '/' + 'neg_index.txt', 'a')
    # svs_index = open(index_path + '/' + 'svs_index.txt', 'a')
    for col in range(address[0]):
        tile = slide_dz.get_tile(dz_level, (col, row))
        flag = check_background(tile, tile_size, ratio, back_threshold)  ##ratio 背景筛选参数
        if flag and col != address[0] - 1 and row != address[1] - 1:
            source_imgage = transform(tile, target_max, target_stain_matrix, 0, 0)  #染色归一化
            tile = Image.fromarray(source_imgage)
            p_mes = out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
            tile.save(out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg')
            pos_index.write(p_mes)
            # cancer_rate = int(np.random.uniform(0, 1) * 255)
            # rate1 = int(np.random.uniform(0, 1) * 255)
            # rate2 = int(np.random.uniform(0, 1) * 255)
            # rate3 = int(np.random.uniform(0, 1) * 255)
            # rate4 = int(np.random.uniform(0, 1) * 255)
            # rate5 = int(np.random.uniform(0, 1) * 255)
            # rate6 = int(np.random.uniform(0, 1) * 255)
            # rate7 = int(np.random.uniform(0, 1) * 255)
            # rate8 = int(np.random.uniform(0, 1) * 255)
            # rate9 = int(np.random.uniform(0, 1) * 255)
            # cancer_type = np.random.randint(1, 10)
            # s_mes = out_path + '/' + ' ' + str(cancer_rate) + ' ' + str(col) + ' ' + \
            #         str(row) + ' ' + str(cancer_type) + \
            #         ' ' + str(rate1) + \
            #         ' ' + str(rate2) + \
            #         ' ' + str(rate3) + \
            #         ' ' + str(rate4) + \
            #         ' ' + str(rate5) + \
            #         ' ' + str(rate6) + \
            #         ' ' + str(rate7) + \
            #         ' ' + str(rate8) + \
            #         ' ' + str(rate9) + \
            #         ' ' + '.jpg' + '\n'

        else:
            # n_mes = out_path + '/' + ' ' + str(col) + ' ' + str(row) + ' ' + '.jpg' + '\n'
            # tile.save(out_path + '/' + str(col) + '_' + str(row) + '.jpg')
            n_mes = out_path + '/' + ' ' + '0' + ' ' + str(col) + ' ' + str(row) + \
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

            # svs_index.write(s_mes)
        continue

    # print('segmentation done')
    ##因多进程的使用无法反向传参只能在保存的list中读取进行信息获取




class tiles_Dataset(data.Dataset):

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.tile_list = glob(root_dir + '/' + '**' + '/*.jpg', recursive=True)

    def __getitem__(self, idx):
        tile_path = self.tile_list[idx]
        tile_PIL = Image.open(tile_path)
        trans_tile = self.transform(tile_PIL)
        # print(trans_tile, tile_path)
        return trans_tile, tile_path

    def __len__(self):
        return len(self.tile_list)

def inference224(model, dataset, device, out_path, index_path):
    model.eval()
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    svs_index = open(index_path + '/' + 'svs_index.txt', 'w')
    neg_index = open(index_path + '/' + 'neg_index.txt', 'r')
    #复制
    # 显示所有源文件内容
    svs_index.write(neg_index.read())
    # print('dataloader',dataloader)
    prob_list = []
    for imgs, path in dataloader:
        # print(imgs)
        imgs = imgs.to(device)
        with torch.no_grad():
            # outputs = model(imgs)
            # probs = F.softmax(outputs, dim=1)
            # labels = torch.argmax(probs, dim=1)
            # batch_log = zip(path, probs.cpu().numpy().tolist(),
            #                 labels.cpu().numpy().tolist())
            outputs, features = model(imgs)
            probs = F.softmax(outputs, dim=1)
            labels = torch.argmax(probs, dim=1)
            batch_log = zip(path, probs.cpu().numpy().tolist(), features.cpu().numpy().tolist(),
                            labels.cpu().numpy().tolist())

            for log in batch_log:
                prob_list.append(log)
                cancer_rate = int(log[1][log[3]] * 255)
                # print(batch_log,log)
                # print(cancer_rate,log[3])
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
    return prob_list

def inference(model, dataset, device, out_path, index_path):
    model.eval()
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    svs_index = open(index_path + '/' + 'svs_index.txt', 'w')
    neg_index = open(index_path + '/' + 'neg_index.txt', 'r')
    #复制
    svs_index.write(neg_index.read())
    prob_list = []
    for imgs, path in dataloader:
        # print(imgs)
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            labels = torch.argmax(probs, dim=1)
            batch_log = zip(path, probs.cpu().numpy().tolist(),
                            labels.cpu().numpy().tolist())
            # outputs, features = model(imgs)
            # probs = F.softmax(outputs, dim=1)
            # labels = torch.argmax(probs, dim=1)
            # batch_log = zip(path, probs.cpu().numpy().tolist(), features.cpu().numpy().tolist(),
            #                 labels.cpu().numpy().tolist())
            # 没有features

            for log in batch_log:
                prob_list.append(log)
                cancer_rate = int(log[1][log[2]] * 255)
                # print(batch_log,log)
                # print(cancer_rate,log[3])
                rate1 = int(log[1][0] * 255)
                rate2 = int(log[1][1] * 255)
                rate3 = 0
                rate4 = 0
                rate5 = 0
                rate6 = 0
                rate7 = 0
                rate8 = 0
                rate9 = 0
                cancer_type = log[2] + 1
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
    return prob_list

# 神经网络输出字典保存
# def write_log(prob_list, file_name, file_path):
#     log = dict()
#     for item in prob_list:
#         log.update({item[0].split('\\')[-1][:-4]: {'probs': item[1], 'features': item[2], 'label': item[3]}})
#     if not os.path.exists(file_path):
#         os.makedirs(file_path)
#     # print(log)
#     print(file_path, file_name)
#     with open(os.path.join(file_path, file_name), 'a') as f:
#         js = json.dumps(log, indent=2)
#         f.write(js)

def net_process(out_path, index_path, tile_size):
    data_transform = transforms.ToTensor()
    print(torch.cuda.current_device(), torch.cuda.get_device_name(0), torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(torch.cuda.current_device(), torch.cuda.get_device_name(0), torch.cuda.is_available())
    else:
        device = torch.device('cpu')

    if tile_size == 224 :
        # pretrained_dict = torch.load('./models/resnet_process_224/best9classtensor_adam0.9985_dropout_32.ckpt') #resnet18 自写的
        pretrained_dict = torch.load('./models/resnet_process_224/train_29_c1_0.9928.ckpt') #resnext50_ 模型体积差一倍
        # model = resnet224()
        # model = models.resnext50_32x4d()
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif tile_size == 112 :
        # pretrained_dict = torch.load('./models/resnet_112_2/train_112_rectified.ckpt')
        pretrained_dict = torch.load('./models/tiny_resnet_112/best9class_tinyresnet112(0.9839).ckpt')
        model = resnet112()
        num_ftrs = model.fc1.in_features
        model.fc1 = nn.Linear(num_ftrs, 9)
    elif tile_size == 56 :
        # pretrained_dict = torch.load('./models/tiny_resnet_56_2/train_46.ckpt')
        pretrained_dict = torch.load('./models/tiny_resnet_56/best9class_tinyresnettensor(0.9664).ckpt')
        model = resnet56()
        num_ftrs = model.fc1.in_features
        model.fc1 = nn.Linear(num_ftrs, 9)
    else:
        return

    new_dict = dict()
    for i in pretrained_dict:
        new_dict.update({i[7:]: pretrained_dict[i]})

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 9)

    # num_ftrs = model.fc1.in_features
    # model.fc1 = nn.Linear(num_ftrs, 9)

    model.load_state_dict(new_dict)
    model.to(device)
    print('Model state loaded')
    dataset = tiles_Dataset(out_path, data_transform) ##整合运行数据
    inference(model, dataset, device, out_path, index_path)
    # print(dataset)
    # if tile_size == 224 :
    #     # prob_list = inference224(model, dataset, device, out_path, index_path)
    #     inference(model, dataset, device, out_path, index_path)
    # else:
        # prob_list = inference(model, dataset, device, svs_name, out_folder, result_folder, index_folder, tile_size)
        # inference(model, dataset, device, out_path, index_path)
    # print('prob', prob_list)
    # print(slide_name,log_dir)
    # write_log(prob_list, slide_name, log_dir)