import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from natsort import natsorted
import numpy as np
import os
import pandas as pd
from point_cloud_utils.data import *
from dataset1_db.parser import read_objects_pose, read_cam_poses, get_cam_pose

def tensor_from_1ch_image(image: np.array) -> torch.Tensor:
    return torch.from_numpy(image).unsqueeze(dim=0)


def tensor_from_nch_image(image: np.array) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1)


def image_from_tensor(tensor: torch.Tensor, div: float = 255.0, scale = True) -> np.ndarray:
    img = tensor.permute(1, 2, 0).numpy()
    if scale:
      img = img * div
    return img.astype(np.float32)

class RealSensorDataset(Dataset):
    def __init__(self, dataset_path, depth_div, transform=None):
        self.transform = transform
        self.dataset_path = dataset_path
        self.inputs = natsorted(os.listdir(self.dataset_path))
        self.depth_div = depth_div
        self.rgb_div = 255.0

    def __getitem__(self, index):
        input_path = os.path.join(self.dataset_path, self.inputs[index])
        input_img = np.load(input_path).astype(np.float32)
        input_rgb = input_img[:, :, 0:3] / self.rgb_div
        input_depth = input_img[:, :, 3:] / self.depth_div
        input_rgb_tensor = tensor_from_nch_image(input_rgb)
        input_depth_tensor = tensor_from_1ch_image(input_depth) if len(
            input_depth.shape) == 2 else tensor_from_nch_image(input_depth)
        input_tensor = torch.cat((input_rgb_tensor, input_depth_tensor), dim=0)
        return input_tensor

    def __len__(self):
        return len(self.inputs)