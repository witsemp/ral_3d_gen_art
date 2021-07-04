import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from point_cloud_utils.data import *
from data_graspable.parser import read_objects_pose, read_cam_poses, get_cam_pose

class FurnitureDataset(Dataset):
    def __init__(self, dataset_path, depth_div, greyscale=False, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.inputs_path = os.path.join(self.dataset_path, 'inputs')
        self.targets_path = os.path.join(self.dataset_path, 'targets')
        self.inputs = natsorted(os.listdir(self.inputs_path))
        self.targets = natsorted(os.listdir(self.targets_path))
        self.depth_div = depth_div
        self.rgb_div = 255.0
        self.greyscale = greyscale
        self.csv_path = os.path.join(self.dataset_path, 'camPoses.txt')
        self.cam_poses = read_cam_poses(self.csv_path)

    def __getitem__(self, index):
        input_path, target_path = os.path.join(self.inputs_path, self.inputs[index]), \
                                  os.path.join(self.targets_path, self.targets[index])

        input_img, target_img = np.load(input_path).astype(np.float32), \
                                np.load(target_path).astype(np.float32) / self.depth_div
        input_rgb = input_img[:, :, 0:3] / self.rgb_div
        input_depth = input_img[:, :, 3:] / self.depth_div
        input_rgb_tensor = tensor_from_nch_image(input_rgb)
        input_depth_tensor = tensor_from_1ch_image(input_depth) if len(
            input_depth.shape) == 2 else tensor_from_nch_image(input_depth)
        target_depth_tensor = tensor_from_1ch_image(target_img)
        input_tensor = torch.cat((input_rgb_tensor, input_depth_tensor), dim=0)
        pose_idx = os.path.splitext(os.path.basename(input_path))[0]
        pose_idx = int(pose_idx.lstrip('0'))
        mat1, mat2 = get_cam_pose(self.cam_poses, pose_idx), get_cam_pose(self.cam_poses, pose_idx + 1)
        return input_tensor, target_depth_tensor, torch.from_numpy(mat1), torch.from_numpy(mat2)

    def __len__(self):
        return len(self.inputs)

class TestDataset(Dataset):
    def __init__(self, dataset_path, depth_div, greyscale=False, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.inputs_path = os.path.join(self.dataset_path, 'inputs')
        self.targets_path = os.path.join(self.dataset_path, 'targets')
        self.inputs = natsorted(os.listdir(self.inputs_path))
        self.targets = natsorted(os.listdir(self.targets_path))
        self.depth_div = depth_div
        self.rgb_div = 255.0
        self.greyscale = greyscale
        self.csv_path = os.path.join(self.dataset_path, 'camPoses.txt')
        self.cam_poses = read_cam_poses(self.csv_path)

    def __getitem__(self, index):
        input_path, target_path = os.path.join(self.inputs_path, self.inputs[index]), \
                                  os.path.join(self.targets_path, self.targets[index])

        input_img, target_img = np.load(input_path).astype(np.float32), \
                                np.load(target_path).astype(np.float32) / self.depth_div
        input_rgb = input_img[:, :, 0:3] / self.rgb_div
        input_depth = input_img[:, :, 3:] / self.depth_div
        input_rgb_tensor = tensor_from_nch_image(input_rgb)
        input_depth_tensor = tensor_from_1ch_image(input_depth) if len(
            input_depth.shape) == 2 else tensor_from_nch_image(input_depth)
        target_depth_tensor = tensor_from_1ch_image(target_img)
        input_tensor = torch.cat((input_rgb_tensor, input_depth_tensor), dim=0)
        pose_idx_text = os.path.splitext(os.path.basename(input_path))[0]
        pose_idx = int(pose_idx_text.strip('0'))
        mat1, mat2 = get_cam_pose(self.cam_poses, pose_idx), get_cam_pose(self.cam_poses, pose_idx + 1)
        return input_tensor, target_depth_tensor, torch.from_numpy(mat1), torch.from_numpy(mat2), pose_idx_text

    def __len__(self):
        return len(self.inputs)