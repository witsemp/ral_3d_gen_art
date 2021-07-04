import cv2 as cv
import natsort
import numpy as np
import open3d.visualization
import torch

from point_cloud_utils.point_cloud import *
from data_graspable.utils import get_rgbdp_paths, listdir_full_path
from data import *
from matplotlib import pyplot as plt
from recgan.voxelize import *
from visualise_results import *

if __name__ == '__main__':
    results_path = '/home/witsemp/Uczelnia/Magisterka/22062021/results.txt'
    data_path = '/home/witsemp/Uczelnia/Magisterka/22062021/UNet_DatasetDB_Furniture_Test'
    gt_data_path = '/home/witsemp/Uczelnia/Magisterka/Data/datasetFurnitures_5categories_5instances'
    transforms_path = '/home/witsemp/Uczelnia/Magisterka/Data/Dataset_DB_Furniture_Test/train/camPoses.txt'

    df = read_prediction_losses(results_path)
    num_samples = 100
    start_idx = 0
    range = visualise_predictions(df, start_idx=start_idx, num_samples=num_samples)

    transforms_df = read_cam_poses(transforms_path)
    depth_div = 5000

    for i, (input_idx, target_idx) in enumerate(range[:5]):
        print(f'Processed {i} out of {len(range) - 1}')

        # Read images
        input_depth = cv.imread(os.path.join(gt_data_path, f'depth{input_idx}.png'), flags=cv.IMREAD_ANYDEPTH)
        input_depth = np.asarray(input_depth, dtype=np.float32) / depth_div
        output_depth = cv.imread(os.path.join(data_path, f'{target_idx}_output.png'), flags=cv.IMREAD_ANYDEPTH)
        output_depth = cv.resize(output_depth, (640, 480), interpolation=cv.INTER_AREA)
        output_depth = np.asarray(output_depth, dtype=np.float32) / depth_div
        target_depth = cv.imread(os.path.join(gt_data_path, f'depth{target_idx}.png'), flags=cv.IMREAD_ANYDEPTH)
        target_depth = np.asarray(target_depth, dtype=np.float32) / depth_div

        # Create point clouds
        input_pcl = pcl_from_images(input_depth, camera_matrix)
        output_pcl = pcl_from_images(output_depth, camera_matrix)
        target_pcl = pcl_from_images(target_depth, camera_matrix)

        # Remove single outliers in ground truth point clouds (cam poses?)
        input_pcl, _ = input_pcl.remove_statistical_outlier(nb_neighbors=2, std_ratio=10)
        # output_pcl, _ = output_pcl.remove_statistical_outlier(nb_neighbors=10, std_ratio=10)
        target_pcl, _ = target_pcl.remove_statistical_outlier(nb_neighbors=2, std_ratio=10)

        # Get transformation matrices
        input_idx_int, target_idx_int = pose_idx_to_in(input_idx), pose_idx_to_in(target_idx)
        mat1, mat2 = get_cam_pose(transforms_df, input_idx_int), get_cam_pose(transforms_df, target_idx_int)

        # Filter point clouds by projection and transform them into common reference frame
        output_pcl = output_pcl.transform(np.linalg.inv(mat1) @ mat2)
        output_pcl = check_occlusion_no_off(input_depth, output_pcl, camera_matrix)
        input_pcl = input_pcl.transform(mat1)
        output_pcl = output_pcl.transform(mat1)
        target_pcl = target_pcl.transform(mat2)

        # Filter out floor and high outliers
        input_pcl.points = open3d.utility.Vector3dVector(
            [point for point in np.asarray(input_pcl.points) if point[2] > 0.05])
        input_points = np.asarray(input_pcl.points)

        max_z_out = np.max(np.asarray(output_pcl.points)[:, 2])
        output_pcl.points = open3d.utility.Vector3dVector(
            [point for point in np.asarray(output_pcl.points) if 0.05 < point[2] < 0.85 * max_z_out])
        output_points = np.asarray(output_pcl.points)

        target_pcl.points = open3d.utility.Vector3dVector(
            [point for point in np.asarray(target_pcl.points) if point[2] > 0.05])
        target_points = np.asarray(target_pcl.points)

        # dst = output_pcl.compute_nearest_neighbor_distance()
        # mean_dst = sum(dst) / len(dst)
        # output_pcl, _ = output_pcl.remove_radius_outlier(10, radius=2 * mean_dst)
        # open3d.visualization.draw_geometries([output_pcl])

        # Create point clouds to voxelize
        point_cloud_predicted, point_cloud_ground_truth = open3d.geometry.PointCloud(), open3d.geometry.PointCloud()
        point_cloud_ground_truth.points = open3d.utility.Vector3dVector(np.concatenate((input_points, target_points)))
        point_cloud_predicted.points = open3d.utility.Vector3dVector(np.concatenate((input_points, output_points)))

        point_cloud_predicted = point_cloud_predicted.paint_uniform_color(np.array([0.0, 1.0, 0.0]))
        point_cloud_ground_truth = point_cloud_ground_truth.paint_uniform_color(np.array([1.0, 0.0, 0.0]))
        # open3d.io.write_point_cloud(f'pcls/gt/{input_idx}.ply', point_cloud_ground_truth)
        # open3d.io.write_point_cloud(f'pcls/predicted/{input_idx}.ply', point_cloud_predicted)
        open3d.visualization.draw_geometries([point_cloud_predicted])

        base_path = 'voxel_grids'
        # input_vox = voxelization(np.asarray(input_pcl.points), save=False, save_name=f'voxel_grids/gan_input/{input_idx}.npz')
        # gt_vox = voxelization_fixed_res(np.asarray(point_cloud_ground_truth.points), save=False, save_name=f'voxel_grids/ground_truth_256/{input_idx}.npz')
        # plotlyFromVoxels(input_vox)
        # plotlyFromVoxels(gt_vox)

        # predicted_vox = voxelization_fixed_res(np.asarray(point_cloud_predicted.points), save_name=f'voxel_grids/predictions_256/{input_idx}.npz')
        # gt_vox = voxelization_fixed_res(np.asarray(point_cloud_ground_truth.points), save_name=f'voxel_grids/ground_truth_256/{input_idx}.npz')
        # input_vox = voxelization(np.asarray(input_pcl.points), save_name=f'voxel_grids/gan_input/{input_idx}.npz')
