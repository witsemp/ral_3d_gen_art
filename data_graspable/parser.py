import pandas as pd
from typing import Tuple, List
import torch
from torch import nn
import numpy as np


def read_cam_poses(cam_poses_path: str) -> pd.DataFrame:
    col_names = ['Pose Index', 'Parent Index', 'Roll', 'Pitch'] + [f'R{row}C{col}' for row in range(4) for col in
                                                                   range(4)]
    return pd.read_csv(cam_poses_path, delim_whitespace=True, names=col_names)


def get_cam_pose(df: pd.DataFrame, idx: int) -> np.ndarray:
    cols = [f'R{i}C{j}' for i in range(4) for j in range(4)]
    mat = df.loc[df['Pose Index'] == idx, :][cols].values[0]
    return np.reshape(mat, newshape=(4, 4))


def pose_idx_to_in(pose_idx: str) -> int:
    return int(pose_idx.lstrip('0')) if pose_idx.lstrip('0') != '' else 0


def read_objects_pose(object_poses_path: str) -> np.ndarray:
    def _list_str_to_int(lst: List[str]) -> List[int]:
        return [int(elem) for elem in lst]

    with open(object_poses_path) as file:
        lines = file.readlines()
    return np.array(_list_str_to_int(lines[-1].rstrip().split(sep=' ')[-2:])).astype(np.float32)


if __name__ == '__main__':
    csv_path = '/home/witsemp/Uczelnia/Magisterka/Data/VisualisationDataset/camPoses.txt'
    poses = read_cam_poses(csv_path)
    mat = get_cam_pose(poses, 10)
    print(mat)
