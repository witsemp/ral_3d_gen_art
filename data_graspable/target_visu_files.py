import os.path
import shutil
from shutil import copy
from utils import *
import numpy as np

def move_target_visualisation_files(file_paths: List[Tuple], dest_path: str):
    scene_rgb_path = os.path.join(dest_path, 'scene_rgb/targets')
    scene_depth_path = os.path.join(dest_path, 'scene_depth/targets')
    dat_path = os.path.join(dest_path, 'dat/targets')
    dirs = [scene_rgb_path, scene_depth_path, dat_path]
    for path in dirs:
        os.makedirs(path, exist_ok=True)
    for paths_tuple in file_paths:
        for i, path in enumerate(paths_tuple):
            shutil.copy(path, dirs[i])

def count_initial_zeros(number: str):
    zero_counter = 0
    for char in number:
        if char == '0':
            zero_counter += 1
        else:
            break
    return zero_counter


visu_dataset_path = '/home/witsemp/Uczelnia/Magisterka/Data/VisualisationDataset'
data_path = '/home/witsemp/Uczelnia/Magisterka/Data/dataset_3categories_10instances'

rgb_input_path = os.path.join(visu_dataset_path, 'scene_rgb/inputs')
all_files = listdir_full_path(data_path)

scene_rgb_targets = []
scene_depth_targets = []
dat_targets = []
for path in listdir_full_path(rgb_input_path):
    pose_idx = os.path.splitext(os.path.basename(path))[0][3:]
    num_zeros = count_initial_zeros(pose_idx)
    numeric_pose_idx = int(pose_idx) if pose_idx[0] != '0' else int(pose_idx.lstrip('0'))
    target_pose_idx = "".join(['0' for i in range(num_zeros)]) + str(numeric_pose_idx + 1)
    rgb_target_path = os.path.join(data_path, f'rgb{target_pose_idx}.png')
    depth_target_path = os.path.join(data_path, f'depth{target_pose_idx}.png')
    dat_target_path = os.path.join(data_path, f'objects{target_pose_idx}.dat')
    if rgb_target_path in all_files and depth_target_path in all_files and dat_target_path in all_files:
        scene_rgb_targets.append(rgb_target_path)
        scene_depth_targets.append(depth_target_path)
        dat_targets.append(dat_target_path)

paths = [scene_rgb_targets, scene_depth_targets, dat_targets]
zipped = list(zip(*paths))


move_target_visualisation_files(zipped, visu_dataset_path)
