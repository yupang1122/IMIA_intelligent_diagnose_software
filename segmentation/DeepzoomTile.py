import openslide
import numpy as np
import os
# import spams as sp
import cv2 as cv
from glob import glob
from PIL import Image
from openslide import deepzoom
from multiprocessing import Pool

def check_background(tile, tile_size, ratio):
    if tile.size != (tile_size, tile_size):
        return False
    gray = tile.convert('L')
    bw = gray.point(lambda x:0 if x< 220 else 1, 'F')
    arr = np.array(np.asarray(bw))
    avgBkg = np.average(bw)
    if avgBkg < ratio:
        return True
    else:
        return False

def get_tiles(slide_name, out_folder, tile_size, magnification, ratio, color_normalization=False):
    slide = openslide.open_slide(slide_name)
    max_mag = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    slide_dz = deepzoom.DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)
    a = np.log2(int(max_mag) / magnification) + 1
    dz_level = int(slide_dz.level_count - a)
    address = slide_dz.level_tiles[int(dz_level)]
    slide_list = glob('E:/Slide_test/*.svs')
    pool =Pool(35)
    pool.map(main, slide_list)
    pool.close()
    pool.join()
    print('poolDone!')

    def task(num):
        print("task%num is running" % num)
        time.sleep(1)
        print("task%num finished" % num)
    if __name__ == "__main__":
        # 创建一个指定进程数的进程池
        pool = Pool(processes=4)
        task_num = [x for x in range(6)]
        start = time.time()
        pool.map(task, task_num)
        pool.close()
        pool.join()
        end = time.time()
        print("All tasks finished in %s seconds" % (end - start))
    for col in range(address[0]):
        for row in range(address[1]):
            tile = slide_dz.get_tile(dz_level, (col, row))
            flag = check_background(tile, tile_size, ratio)
            svs_name = slide_name.split('/')[-1][:-4]
            out_path = os.path.join(out_folder, svs_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            if flag:
                if color_normalization:
                    # source_imgage = transform(tile, target_max, target_stain_matrix, 0, 0)
                    source_imgage = transform(tile, 0, 0)
                    tile = Image.fromarray(source_imgage)
                tile.save(out_path + '/' + str(col) + '_' + str(row) + '.jpg')
            else:
                continue
def tilesave(slide_dz, dz_level, col, row, tile_size, ratio, out_path):
    tile = slide_dz.get_tile(dz_level, (col, row))
    flag = check_background(tile, tile_size, ratio)

    if flag:
        tile.save(out_path + '/' + str(col) + '_' + str(row) + '.jpg')
    else:
        return

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

def main(slide_name):
    out_folder = 'E:/Code/HistoSlider/histoslider/dataprocess'
    print(slide_name)
    get_tiles(slide_name, out_folder, 224, 20, 0.5, False)

if __name__ == '__main__':
    # target_max, target_stain_matrix = get_target_max('/media/dell-workstation/data8T/Blackz/data_preprocess/image/50_13.jpg')
    slide_list = glob('E:/Slide_test/*.svs')
    pool =Pool(35)
    pool.map(main, slide_list)
    pool.close()
    pool.join()
    print('poolDone!')
