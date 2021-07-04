import os
import cv2 as cv
import numpy as np
import re
from natsort import natsorted
from shutil import copy
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
from sklearn.model_selection import train_test_split

def listdir_full_path(path: str) -> List[str]:
    return [os.path.join(path, p) for p in os.listdir(path)]


def get_rgbdp_paths(path: str) -> Tuple[List, List, List, List]:
    rgb_input_paths = []
    depth_input_paths = []
    depth_target_paths = []
    projected_paths = []
    for image_path in listdir_full_path(path):
        if re.match('.*rgbIn.*yaw0', image_path):
            rgb_input_paths.append(image_path)
        elif re.match('.*depthIn.*yaw0', image_path):
            depth_input_paths.append(image_path)
        elif re.match('.*depthIn.*yaw-179', image_path):
            depth_target_paths.append(image_path)
        elif re.match('.*depthProj.*yaw-179', image_path):
            projected_paths.append(image_path)
    return (natsorted(rgb_input_paths),
            natsorted(depth_input_paths),
            natsorted(depth_target_paths),
            natsorted(projected_paths))

def get_rgbdpd_paths(path: str) -> Tuple[List, List, List, List, List]:
    rgb_input_paths = []
    depth_input_paths = []
    depth_target_paths = []
    projected_paths = []
    distance_paths = []
    for image_path in listdir_full_path(path):
        if re.match('.*rgbIn.*yaw0', image_path):
            rgb_input_paths.append(image_path)
        elif re.match('.*depthIn.*yaw0', image_path):
            depth_input_paths.append(image_path)
        elif re.match('.*depthIn.*yaw-179', image_path):
            depth_target_paths.append(image_path)
        elif re.match('.*depthProj.*yaw-179', image_path):
            projected_paths.append(image_path)
        elif re.match('.*distIn.*yaw0', image_path):
            distance_paths.append(image_path)
    return (natsorted(rgb_input_paths),
            natsorted(depth_input_paths),
            natsorted(depth_target_paths),
            natsorted(projected_paths),
            natsorted(distance_paths))

def make_npy(input_path: str, dest_path):
    input_dest_path = os.path.join(dest_path, 'input')
    target_dest_path = os.path.join(dest_path, 'target')
    os.makedirs(input_dest_path, exist_ok=True)
    os.makedirs(target_dest_path, exist_ok=True)
    paths = get_rgbdp_paths(input_path)
    for i, obj in enumerate(list(zip(*paths))):
        if not i % 100:
            print(f"Saved {i} out of {len(list(zip(*paths))) - 1}")
        # obj[0] - input rgb, obj[1] - input depth, obj[2] - target_depth, obj[3] - projected
        input_rgb_img = cv.cvtColor(cv.imread(obj[0], flags=cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        input_depth_img = cv.imread(obj[1], flags=cv.IMREAD_ANYDEPTH)[..., None]
        target_depth_img = cv.imread(obj[2], flags=cv.IMREAD_ANYDEPTH)
        projected_img = cv.imread(obj[3], flags=cv.IMREAD_ANYDEPTH)[..., None]
        stacked = np.concatenate((input_rgb_img, input_depth_img, projected_img), axis=2)
        basename = os.path.basename(obj[0])
        save_name = basename[basename.find('n') + 1:basename.find('y')] + basename[
                                                                          basename.find('w') + 1:basename.find(
                                                                              '.')] + '.npy'
        np.save(os.path.join(input_dest_path, save_name), stacked)
        np.save(os.path.join(target_dest_path, save_name), target_depth_img)

def make_npy(paths: List[List[str]], dest_path: str):
    inputs_dest_path = os.path.join(dest_path, 'input')
    targets_dest_path = os.path.join(dest_path, 'target')
    os.makedirs(inputs_dest_path, exist_ok=True)
    os.makedirs(targets_dest_path, exist_ok=True)
    for i, obj in enumerate(list(zip(*paths))):
        if not i % 100:
            print(f"Saved {i} out of {len(list(zip(*paths))) - 1}")
        input_rgb_img = cv.cvtColor(cv.imread(obj[0], flags=cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        input_depth_img = cv.imread(obj[1], flags=cv.IMREAD_ANYDEPTH)[..., None]
        target_depth_img = cv.imread(obj[2], flags=cv.IMREAD_ANYDEPTH)
        projected_depth_img = cv.imread(obj[3], flags=cv.IMREAD_ANYDEPTH)[..., None]
        stacked = np.concatenate((input_rgb_img, input_depth_img, projected_depth_img), axis=2)
        basename = os.path.basename(obj[0])
        save_name = basename[basename.find('n') + 1:basename.find('y')] + basename[
                                                                          basename.find('w') + 1:basename.find(
                                                                              '.')] + '.npy'
        np.save(os.path.join(inputs_dest_path, save_name), stacked)
        np.save(os.path.join(targets_dest_path, save_name), target_depth_img)

def make_npy_distance(paths: List[List[str]], dest_path: str):
    inputs_dest_path = os.path.join(dest_path, 'input')
    targets_dest_path = os.path.join(dest_path, 'target')
    os.makedirs(inputs_dest_path, exist_ok=True)
    os.makedirs(targets_dest_path, exist_ok=True)
    for i, obj in enumerate(list(zip(*paths))):
        if not i % 100:
            print(f"Saved {i} out of {len(list(zip(*paths))) - 1}")
        input_rgb_img = cv.cvtColor(cv.imread(obj[0], flags=cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
        input_depth_img = cv.imread(obj[1], flags=cv.IMREAD_ANYDEPTH)[..., None]
        target_depth_img = cv.imread(obj[2], flags=cv.IMREAD_ANYDEPTH)
        projected_depth_img = cv.imread(obj[3], flags=cv.IMREAD_ANYDEPTH)[..., None]
        distance_img = cv.imread(obj[4], cv.IMREAD_UNCHANGED)[..., None]
        stacked = np.concatenate((input_rgb_img, input_depth_img, projected_depth_img, distance_img), axis=2)
        basename = os.path.basename(obj[0])
        save_name = basename[basename.find('n') + 1:basename.find('y')] + basename[
                                                                          basename.find('w') + 1:basename.find(
                                                                              '.')] + '.npy'
        np.save(os.path.join(inputs_dest_path, save_name), stacked)
        np.save(os.path.join(targets_dest_path, save_name), target_depth_img)


def move_inputs_targets(inputs_list: List,
                        targets_list: List,
                        target_dir_path: str):
    for i, (input_name, target_name) in enumerate(zip(inputs_list, targets_list)):
        copy(input_name, os.path.join(target_dir_path, 'inputs'))
        copy(target_name, os.path.join(target_dir_path, 'targets'))


def make_dataset(dataset_path):
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    try:
        for path in [train_path, valid_path, test_path]:
            inputs_path = os.path.join(path, 'inputs')
            targets_path = os.path.join(path, 'targets')
            os.mkdir(path)
            os.mkdir(inputs_path)
            os.mkdir(targets_path)
    except OSError:
        print("Train/valid directories creation failed")