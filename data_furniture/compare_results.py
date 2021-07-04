import os

import natsort
import numpy as np
from recgan.voxelize import plotlyFromVoxels
from data_graspable.utils import listdir_full_path
from scipy import io

def iou(gt_voxel_grid: np.ndarray, predicted_voxel_grid: np.ndarray) -> float:
    gt_voxel_grid_bool = gt_voxel_grid.astype(bool)
    predicted_voxel_grid_bool = predicted_voxel_grid.astype(bool)
    overlap = gt_voxel_grid_bool * predicted_voxel_grid_bool
    union = gt_voxel_grid_bool + predicted_voxel_grid_bool
    val = overlap.sum() / float(union.sum())

if __name__ == '__main__':

    base_path = 'voxel_grids'
    gt_path = os.path.join(base_path, 'ground_truth_256')
    gan_input_path = os.path.join(base_path, 'gan_input')
    gan_output_path = os.path.join(base_path, 'gan_output')
    ours_output_path = os.path.join(base_path, 'predictions_256')

    gt_grid_paths = natsort.natsorted(listdir_full_path(gt_path))
    gan_grid_paths = natsort.natsorted(listdir_full_path(gan_output_path))
    gan_input_grid_paths = natsort.natsorted(listdir_full_path(gan_input_path))
    ours_grid_paths = natsort.natsorted(listdir_full_path(ours_output_path))
    idx = 0
    gt_grid = np.load(gt_grid_paths[idx])['arr_0']
    ours_output_grid = np.load(ours_grid_paths[idx])['arr_0']

    gan_input_grid = np.load(gan_input_grid_paths[idx])['arr_0']
    gan_output_grid = io.loadmat(gan_grid_paths[idx])['Y_test_pred']

    th = 0.5
    gan_output_grid[gan_output_grid >= th] = 1
    gan_output_grid[gan_output_grid < th] = 0


    iou(gt_grid, ours_output_grid)
    iou(gt_grid, gan_output_grid)
    plotlyFromVoxels(gt_grid)
    plotlyFromVoxels(gan_input_grid)
    plotlyFromVoxels(ours_output_grid)
    plotlyFromVoxels(gan_output_grid)