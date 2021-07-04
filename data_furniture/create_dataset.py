import os
import re
from natsort import natsorted
from sklearn.model_selection import train_test_split
from data_graspable.utils import listdir_full_path, make_dataset
from typing import List
import cv2 as cv
import numpy as np


def get_paths(dir_path: str):
    all_paths = listdir_full_path(dir_path)
    rgb, depth, projected = [], [], []
    for i, path in enumerate(all_paths):
        fname_wo_ext = os.path.splitext(os.path.basename(path))[0]
        if re.match('^depth[0-9]+$', fname_wo_ext):
            depth.append(path)
        elif re.match('^rgb[0-9]+$', fname_wo_ext):
            rgb.append(path)
        elif re.match('^depthProj[0-9]+$', fname_wo_ext):
            projected.append(path)
    return [natsorted(rgb), natsorted(depth), natsorted(projected)]


def split_paths(paths: List):
    rgb, depth, projected = paths
    rgb_in, depth_in, depth_target = [], [], []
    for i in range(len(rgb)):
        if not i % 2:
            rgb_in.append(rgb[i])
            depth_in.append(depth[i])
        else:
            depth_target.append(depth[i])
    return list(zip(rgb_in, depth_in, projected)), depth_target


def manual_resize(image, dst_shape):
    h_in, w_in = image.shape
    h_dst, w_dst = dst_shape
    h_range = list(range(h_in))
    w_range = list(range(w_in))
    h_stride = h_in // h_dst
    w_stride = w_in // w_dst
    dst_image = np.zeros(dst_shape)
    for i, row in enumerate(h_range[0::h_stride]):
        for j, col in enumerate(w_range[0::w_stride]):
            dst_image[i, j] = image[row, col]
    return dst_image.astype(np.float32)


def move_files(inputs_list: List,
               targets_list: List,
               target_dir_path: str):
    dest_shape = (120, 160)
    for i, (inputs, target) in enumerate(zip(inputs_list, targets_list)):
        if not i % 100:
            print(f'Saved {i} out of {len(inputs_list) - 1}')
        input_idx = os.path.splitext(os.path.basename(inputs[0]))[0][3:]
        target_idx = os.path.splitext(os.path.basename(target))[0][5:]
        rgb = cv.cvtColor(cv.imread(inputs[0]), code=cv.COLOR_BGR2RGB)
        depth_in = cv.imread(inputs[1], flags=cv.IMREAD_ANYDEPTH)[..., None]
        projected = cv.imread(inputs[2], flags=cv.IMREAD_ANYDEPTH)[..., None]
        depth_target = cv.imread(target, flags=cv.IMREAD_ANYDEPTH)[..., None]
        rgb = cv.resize(rgb, (160, 120))
        depth_in = manual_resize(depth_in, dest_shape)[..., None]
        projected = manual_resize(projected, dest_shape)[..., None]
        depth_target = manual_resize(depth_target, dest_shape)
        npy = np.concatenate((rgb, depth_in, projected), axis=2)
        np.save(os.path.join(target_dir_path, f'inputs/{input_idx}.npy'), npy)
        np.save(os.path.join(target_dir_path, f'targets/{target_idx}.npy'), depth_target)


data_path = '/home/witsemp/Uczelnia/Magisterka/Data/datasetFurnitures_3categories_10instances'
paths = get_paths(data_path)
inputs, targets = split_paths(paths)
print(len(inputs))
print(len(targets))
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                          targets,
                                                                          test_size=0.1,
                                                                          random_state=42,
                                                                          shuffle=True)

inputs_valid, inputs_test, targets_valid, targets_test = train_test_split(inputs_test,
                                                                          targets_test,
                                                                          test_size=0.5,
                                                                          random_state=42,
                                                                          shuffle=True)



dest_path = '/home/witsemp/Uczelnia/Magisterka/Data/Dataset_DB_Furniture_Full_Res'
make_dataset(dest_path)
train_path = os.path.join(dest_path, 'train')
valid_path = os.path.join(dest_path, 'valid')
test_path = os.path.join(dest_path, 'test')
move_files(inputs_train, targets_train, train_path)
move_files(inputs_valid, targets_valid, valid_path)
move_files(inputs_test, targets_test, test_path)
