from dataset_db.utils import listdir_full_path
from point_cloud_utils.point_cloud import manual_resize
import natsort
import os
import cv2 as cv
import numpy as np



path_orig = '/home/witsemp/Uczelnia/Magisterka/RealDataTest/01845'
path_projected = 'projected'
path_npy = 'real_data_npy'

rgbs = natsort.natsorted(
    [file for file in listdir_full_path(path_orig) if os.path.splitext(os.path.basename(file))[1] == '.jpg'])
depths = natsort.natsorted(
    [file for file in listdir_full_path(path_orig) if os.path.splitext(os.path.basename(file))[1] == '.png'])
projected = natsort.natsorted(listdir_full_path(path_projected))
dest_shape = (120, 160)
for (rgb_path, depth_path, proj_path) in list(zip(rgbs, depths, projected)):
    rgb_img = cv.cvtColor(cv.imread(rgb_path, flags=cv.IMREAD_COLOR), code=cv.COLOR_BGR2RGB)
    depth_img = cv.imread(depth_path, flags=cv.IMREAD_ANYDEPTH).astype(np.float32)
    proj_img = cv.imread(proj_path, flags=cv.IMREAD_ANYDEPTH).astype(np.float32)

    rgb_img = cv.resize(rgb_img, (160, 120))
    depth_img = manual_resize(depth_img, dest_shape)[..., None]
    proj_img = manual_resize(proj_img, dest_shape)[..., None]
    # print(np.unique(proj_img))

    npy = np.concatenate((rgb_img, depth_img, proj_img), axis=2)
    file_name = f'{os.path.join(path_npy, os.path.splitext(os.path.basename(depth_path))[0])}.npy'
    np.save(file_name, npy)