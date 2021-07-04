import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from point_cloud_utils.data import *
from parser import read_objects_pose, read_cam_poses, get_cam_pose


class RotationDataset(Dataset):
    def __init__(self, dataset_path, object_locs_path, depth_div, greyscale=False, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.object_locs_path = object_locs_path
        self.inputs_path = os.path.join(self.dataset_path, 'inputs')
        self.targets_path = os.path.join(self.dataset_path, 'targets')
        self.inputs = natsorted(os.listdir(self.inputs_path))
        self.targets = natsorted(os.listdir(self.targets_path))
        self.depth_div = depth_div
        self.rgb_div = 255.0
        self.greyscale = greyscale

    def __getitem__(self, index):
        input_path, target_path = os.path.join(self.inputs_path, self.inputs[index]), \
                                  os.path.join(self.targets_path, self.targets[index])

        input_img, target_img = np.load(input_path).astype(np.float32), \
                                np.load(target_path).astype(np.float32) / self.depth_div
        input_rgb = input_img[:, :, 0:3] / self.rgb_div
        input_grey = cv.cvtColor(input_rgb, cv.COLOR_RGB2GRAY)
        input_depth = input_img[:, :, 3:] / self.depth_div
        input_rgb_tensor = tensor_from_nch_image(input_rgb)
        input_depth_tensor = tensor_from_1ch_image(input_depth) if len(
            input_depth.shape) == 2 else tensor_from_nch_image(input_depth)
        target_depth_tensor = tensor_from_1ch_image(target_img)
        if self.greyscale:
            input_grey_tensor = tensor_from_1ch_image(input_grey)
            input_tensor = torch.cat((input_depth_tensor, input_grey_tensor), dim=0)
        else:
            input_tensor = torch.cat((input_rgb_tensor, input_depth_tensor), dim=0)

        return input_tensor, target_depth_tensor

    def __len__(self):
        return len(self.inputs)


class VisualisationDataset(Dataset):
    def __init__(self, dataset_path, depth_div, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.inputs_path = os.path.join(self.dataset_path, 'npy/inputs')
        self.targets_path = os.path.join(self.dataset_path, 'npy/targets')

        self.scene_rgb_inputs_path = os.path.join(self.dataset_path, 'scene_rgb/inputs')
        self.scene_depth_inputs_path = os.path.join(self.dataset_path, 'scene_depth/inputs')
        self.scene_rgb_targets_path = os.path.join(self.dataset_path, 'scene_rgb/targets')
        self.scene_depth_targets_path = os.path.join(self.dataset_path, 'scene_depth/targets')

        self.dat_inputs_path = os.path.join(self.dataset_path, 'dat/inputs')
        self.dat_targets_path = os.path.join(self.dataset_path, 'dat/targets')

        self.csv_path = os.path.join(self.dataset_path, 'camPoses.txt')

        self.inputs = natsorted(os.listdir(self.inputs_path))
        self.targets = natsorted(os.listdir(self.targets_path))

        self.scene_rgb_input_imgs = natsorted(os.listdir(self.scene_rgb_inputs_path))
        self.scene_depth_input_imgs = natsorted(os.listdir(self.scene_depth_inputs_path))
        self.scene_rgb_target_imgs = natsorted(os.listdir(self.scene_rgb_targets_path))
        self.scene_depth_target_imgs = natsorted(os.listdir(self.scene_depth_targets_path))

        self.input_dats = natsorted(os.listdir(self.dat_inputs_path))
        self.target_dats = natsorted(os.listdir(self.dat_targets_path))

        self.cam_poses = read_cam_poses(self.csv_path)
        self.depth_div = depth_div
        self.rgb_div = 255.0

    def __getitem__(self, index):
        input_path, target_path = os.path.join(self.inputs_path, self.inputs[index]), \
                                  os.path.join(self.targets_path, self.targets[index])
        rgb_input_path, depth_input_path = os.path.join(self.scene_rgb_inputs_path, self.scene_rgb_input_imgs[index]), \
                                           os.path.join(self.scene_depth_inputs_path,
                                                        self.scene_depth_input_imgs[index])

        rgb_target_path, depth_target_path = os.path.join(self.scene_rgb_targets_path,
                                                          self.scene_rgb_target_imgs[index]), \
                                             os.path.join(self.scene_depth_targets_path,
                                                          self.scene_depth_target_imgs[index])

        object_input_pose_path = os.path.join(self.dat_inputs_path, self.input_dats[index])
        object_target_pose_path = os.path.join(self.dat_targets_path, self.target_dats[index])

        input_img, target_img = np.load(input_path).astype(np.float32), \
                                np.load(target_path).astype(np.float32) / self.depth_div
        input_rgb = input_img[:, :, 0:3] / self.rgb_div
        input_depth = input_img[:, :, 3:] / self.depth_div
        input_rgb_tensor = tensor_from_nch_image(input_rgb)
        input_depth_tensor = tensor_from_1ch_image(input_depth) if len(
            input_depth.shape) == 2 else tensor_from_nch_image(input_depth)
        target_depth_tensor = tensor_from_1ch_image(target_img)
        input_tensor = torch.cat((input_rgb_tensor, input_depth_tensor), dim=0)

        scene_rgb, scene_depth = cv.imread(rgb_input_path, flags=cv.IMREAD_COLOR), cv.imread(
            depth_input_path, flags=cv.IMREAD_ANYDEPTH)
        target_scene_rgb, target_scene_depth = cv.imread(rgb_target_path, flags=cv.IMREAD_COLOR), cv.imread(
            depth_target_path, flags=cv.IMREAD_ANYDEPTH)

        object_input_loc = read_objects_pose(object_input_pose_path)
        object_target_loc = read_objects_pose(object_target_pose_path)

        pose_idx = os.path.splitext(os.path.basename(input_path))[0]
        pose_idx = int(pose_idx[:pose_idx.find('o')].strip('0'))
        mat1, mat2 = get_cam_pose(self.cam_poses, pose_idx), get_cam_pose(self.cam_poses, pose_idx + 1)

        return input_tensor, \
               target_depth_tensor, \
               torch.from_numpy(scene_rgb), \
               torch.from_numpy(scene_depth.astype(np.float32)), \
               torch.from_numpy(target_scene_rgb), \
               torch.from_numpy(target_scene_depth.astype(np.float32)), \
               torch.from_numpy(object_input_loc), \
               torch.from_numpy(object_target_loc), \
               torch.from_numpy(mat1), \
               torch.from_numpy(mat2)

    def __len__(self):
        return len(self.inputs)


class LocationDataset(Dataset):
    def __init__(self, dataset_path, object_locs_path, depth_div, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.object_locs_path = object_locs_path
        self.inputs_path = os.path.join(self.dataset_path, 'inputs')
        self.targets_path = os.path.join(self.dataset_path, 'targets')
        self.inputs = natsorted(os.listdir(self.inputs_path))
        self.targets = natsorted(os.listdir(self.targets_path))
        self.depth_div = depth_div
        self.rgb_div = 255.0

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

        pose_idx = os.path.splitext(os.path.basename(input_path))[0][:5]
        dat_file = os.path.join(self.object_locs_path, f'objects{pose_idx}.dat')
        object_pose = read_objects_pose(dat_file)

        return input_tensor, target_depth_tensor, torch.from_numpy(object_pose)

    def __len__(self):
        return len(self.inputs)


if __name__ == '__main__':
    path = '/home/witsemp/Uczelnia/Magisterka/Data/Dataset_DB_3C_10IN/train'
    locs_path = '/home/witsemp/Uczelnia/Magisterka/Data/Dataset_DB_3C_10IN/locs'
    location_dataset = LocationDataset(dataset_path=path, object_locs_path=locs_path, depth_div=30562, transform=None)
    location_dataloader = DataLoader(dataset=location_dataset, batch_size=1, shuffle=True)
    next(iter(location_dataloader))
