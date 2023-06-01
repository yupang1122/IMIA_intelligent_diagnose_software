import pickle
# import unet as U_Net
from network import U_Net
import torch
from PIL import Image
from torchvision import transforms as T
import random
import numpy as np
from torchvision.transforms import ToPILImage
from torchvision.transforms import PILToTensor
import os
from tissueloc.locate_tissue import remove_small_tissue, find_tissue_cnts
import cv2
from skimage import io, color
from torchvision import utils as vutils


def load_model(PATH):
    model_state_dict = torch.load(PATH)
    model = U_Net()
    model.load_state_dict(model_state_dict)
    return model


def initial(result_image, threshold=0.1):
    result_image = result_image > threshold
    result_image = result_image + 0.0
    return result_image


def image_crop(image):  # 图像预处理

    Transform = []
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = Transform(image)  # 转换为TENSOR
    Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = Norm_(image)  # 图像归一化
    return image


def background_net_process():
    tensor_to_pil = ToPILImage()  # tensor -> img
    pil_to_tensor = PILToTensor()
    u_net_model = load_model('E:/Code/HistoSlider/histoslider/segmentation/models/U_Net-50-0.0003-70-0.4000.pkl')

    root = 'E:/Code/HistoSlider/histoslider/segmentation/use_model/target_image'
    image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

    image_path = image_paths[1]
    image = Image.open(image_path)
    pre_image = image_crop(image)

    image = pre_image.resize(1, 3, 1024, 1024)
    prediction = u_net_model(image)

    result_image = initial(prediction)

    result_image = result_image.resize(1, 1024, 1024)
    result = tensor_to_pil(result_image)

    result.save('./use_model/result/result_%d.jpg' % 1)

    cnts = find_tissue_cnts(result)
    slide_img = io.imread(image_path)
    slide_img = np.ascontiguousarray(slide_img, dtype=np.uint8)
    cv2.drawContours(slide_img, cnts, -1, (0, 255, 0), 3)
    filename = 'result_' + str(1) + '_cnt.jpg'

    io.imsave(os.path.join('./use_model/result', filename), slide_img)

    print("we used ", image_paths[1])
