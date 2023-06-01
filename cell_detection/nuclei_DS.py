import os
import numpy as np
import cv2
from time import time
from cell_detection.util import *
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def unet_process( img, ):
    patch_size = 128
    stride = 16

    model_name = 'nucles_model_v3.meta'
    model_path = os.path.join('./cell_detection/models/')
    # print(model_path)
    model = restored_model(os.path.join(model_path, model_name), model_path)
    # result_path=os.path.join(temp_path, 'mask.png')
    # temp_image = cv2.imread(os.path.join(temp_path, temp_name+format))
    temp_image = img

    batch_group, shape = preprocess(temp_image, patch_size, stride)
    mask_list = sess_interference(model, batch_group)
    # print('numbers', format(np.size(mask_list)), len(mask_list))

    # print(mask_list)
    c_mask = patch2image(mask_list, patch_size, stride, shape)

    c_mask = cv2.medianBlur((255 * c_mask).astype(np.uint8), 3)
    c_mask = c_mask.astype(np.float) / 255
    thr = 0.5
    c_mask[c_mask < thr] = 0
    c_mask[c_mask >= thr] = 1
    # print(c_mask)
    center_edge_mask, gray_map = center_edge(c_mask, temp_image)
    # x, y = np.where(gray_map != 0)
    # mask_point = []
    # for i in range(len(x)):
    #     c = []
    #     pointx = int(y[i] + a)
    #     pointy = int(x[i] + b)
    #     c.append(pointx)
    #     c.append(pointy)
    #     mask_point.append(c)
    # print('numbers', format(np.size(c_mask)))
    # print('point', mask_point)

    contours, hirarchy = cv2.findContours(gray_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # print('numbers', format(np.size(contours)))
    # cv2.imwrite(os.path.join('./cell_detection', 'mask.png'), gray_map)
    # cv2.imwrite(os.path.join('./cell_detection', 'label.png'), center_edge_mask)

    #
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(cv2.cvtColor(center_edge_mask, cv2.COLOR_BGR2RGB))
    # ax[0].set_title('label')
    # ax[1].imshow(gray_map)
    # ax[1].set_title('Center and contour')


    model.close_sess()
    print('mask generation done')
    # plt.show()
    return contours

def unet_tile_process(img,):
    patch_size = 128
    stride = 16
    model_name = 'nucles_model_v3.meta'
    model_path = os.path.join('./cell_detection/models/')
    print(model_path)
    model = restored_model(os.path.join(model_path, model_name), model_path)
    # result_path=os.path.join(temp_path, 'mask.png')
    # temp_image = cv2.imread(os.path.join(temp_path, temp_name+format))
    temp_image = img

    batch_group, shape = preprocess(temp_image, patch_size, stride)
    mask_list = sess_interference(model, batch_group)
    # print(mask_list)
    c_mask = patch2image(mask_list, patch_size, stride, shape)
    c_mask = cv2.medianBlur((255 * c_mask).astype(np.uint8), 3)
    c_mask = c_mask.astype(np.float) / 255
    thr = 0.5
    c_mask[c_mask < thr] = 0
    c_mask[c_mask >= thr] = 1
    # print(c_mask)
    # center_edge_mask, gray_map = center_edge(c_mask, temp_image)
    # x, y = np.where(gray_map != 0)
    # mask_point = []
    # for i in range(len(x)):
    #     c = []
    #     pointx = int(y[i] + a)
    #     pointy = int(x[i] + b)
    #     c.append(pointx)
    #     c.append(pointy)
    #     mask_point.append(c)
    print('numbers', format(np.size(c_mask)))
    # print('point', mask_point)



    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(cv2.cvtColor(center_edge_mask, cv2.COLOR_BGR2RGB))
    # ax[0].set_title('label')
    # ax[1].imshow(gray_map)
    # ax[1].set_title('Center and contour')


    model.close_sess()
    print('mask generation done')
    # plt.show()
    return