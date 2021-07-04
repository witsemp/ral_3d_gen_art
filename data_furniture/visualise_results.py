import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import os
from point_cloud_utils.point_cloud import *
from data_graspable.parser import *
import time


def read_prediction_losses(path: str) -> pd.DataFrame:
    predictions_df = pd.read_csv(path, sep=';', names=['idx_input', 'idx_target', 'loss'], dtype={'idx_input': str,
                                                                                                  'idx_target': str,
                                                                                                  'loss': np.float64})
    predictions_df = predictions_df.sort_values(by='loss').drop_duplicates()
    predictions_df = predictions_df.reset_index(drop=True)
    return predictions_df


def visualise_predictions(df: pd.DataFrame, start_idx: int, num_samples=10):
    return list(zip(df.loc[start_idx:start_idx + num_samples - 1, 'idx_input'].values,
                    df.loc[start_idx:start_idx + num_samples - 1, 'idx_target'].values))


def visualiser(geometries: List, transform_mat: np.ndarray, sufix: str):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    for pcl in geometries:
        vis.add_geometry(pcl)
    view_ctl = vis.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = np.linalg.inv(transform_mat)
    view_ctl.convert_from_pinhole_camera_parameters(cam)
    vis.run()
    vis.capture_screen_image(f"reconstructions/0_5/{input_idx}_{sufix}.png")
    vis.destroy_window()
    del view_ctl
    del cam
    del vis

def visualiser_static(geometries: List, transform_mat: np.ndarray):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    for pcl in geometries:
        vis.add_geometry(pcl)
    view_ctl = vis.get_view_control()
    cam = view_ctl.convert_to_pinhole_camera_parameters()
    cam.extrinsic = transform_mat
    view_ctl.convert_from_pinhole_camera_parameters(cam)
    vis.run()
    vis.destroy_window()
    del view_ctl
    del cam
    del vis

if __name__ == '__main__':


    results_path = '/home/witsemp/Uczelnia/Magisterka/22062021/results.txt'
    data_path = '/home/witsemp/Uczelnia/Magisterka/22062021/UNet_DatasetDB_Furniture_Test'
    df = read_prediction_losses(results_path)
    num_samples = 5
    start_idx = 800
    depth_div = 5000
    range = visualise_predictions(df, start_idx=start_idx, num_samples=num_samples)

    show_plt = False
    if show_plt:
        fig, ax = plt.subplots(nrows=num_samples, ncols=5, figsize=(30, 30))
        title = f'Dataset Furniture - Test - {start_idx} - {start_idx + num_samples - 1}'
        col_names = ['RGB', 'Depth', 'Projected', 'Target', 'Output']
        fig.suptitle(title, fontsize=16)
        for i, a in enumerate(ax[0]):
            a.set_title(col_names[i])

        for i, (input_idx, target_idx) in enumerate(range):
            input_rgb_img = cv.imread(os.path.join(data_path, f'{input_idx}_rgb.png'), flags=cv.IMREAD_COLOR)
            input_depth_img = cv.imread(os.path.join(data_path, f'{input_idx}_depth.png'), flags=cv.IMREAD_ANYDEPTH)
            projected_depth_img = cv.imread(os.path.join(data_path, f'{input_idx}_projected.png'),
                                            flags=cv.IMREAD_ANYDEPTH)
            target_img = cv.imread(os.path.join(data_path, f'{target_idx}_target.png'), flags=cv.IMREAD_ANYDEPTH)
            output_img = cv.imread(os.path.join(data_path, f'{target_idx}_output.png'), flags=cv.IMREAD_ANYDEPTH)
            ax[i][0].imshow(input_rgb_img.astype(np.uint8))
            ax[i][1].imshow(input_depth_img.astype(float))
            ax[i][2].imshow(projected_depth_img.astype(float))
            ax[i][3].imshow(target_img.astype(float))
            ax[i][4].imshow(output_img.astype(float))
        plt.show()

    show_o3d = True
    transforms_path = '/home/witsemp/Uczelnia/Magisterka/Data/Dataset_DB_Furniture_Test/train/camPoses.txt'
    gt_path = '/home/witsemp/Uczelnia/Magisterka/Data/datasetFurnitures_5categories_5instances'
    transforms_df = read_cam_poses(transforms_path)

    if show_o3d:
        for i, (input_idx, target_idx) in enumerate(range):
            input_rgb = cv.imread(os.path.join(data_path, f'{input_idx}_rgb.png'), flags=cv.IMREAD_COLOR)
            input_rgb = cv.resize(input_rgb, (640, 480))

            target_rgb = cv.imread(os.path.join(gt_path, f'rgb{target_idx}.png'), flags=cv.IMREAD_COLOR)

            input_depth = cv.imread(os.path.join(data_path, f'{input_idx}_depth.png'), flags=cv.IMREAD_ANYDEPTH)
            input_depth = cv.resize(input_depth, (640, 480), interpolation=cv.INTER_AREA)
            input_depth = np.asarray(input_depth, dtype=np.float32) / depth_div

            output_depth = cv.imread(os.path.join(data_path, f'{target_idx}_output.png'), flags=cv.IMREAD_ANYDEPTH)
            output_depth = cv.resize(output_depth, (640, 480), interpolation=cv.INTER_AREA)
            output_depth = np.asarray(output_depth, dtype=np.float32) / depth_div

            target_depth = cv.imread(os.path.join(data_path, f'{target_idx}_target.png'), flags=cv.IMREAD_ANYDEPTH)
            target_depth = cv.resize(target_depth, (640, 480), interpolation=cv.INTER_AREA)
            target_depth = np.asarray(target_depth, dtype=np.float32) / depth_div

            input_pcl = pcl_from_rgbd(input_depth, input_rgb.astype(np.uint8), camera_matrix_opencv)
            output_pcl = pcl_from_images(output_depth, camera_matrix)
            target_pcl = pcl_from_rgbd(target_depth, target_rgb, camera_matrix_opencv)

            input_idx_int, target_idx_int = pose_idx_to_in(input_idx), pose_idx_to_in(target_idx)

            mat1, mat2 = get_cam_pose(transforms_df, input_idx_int), get_cam_pose(transforms_df, target_idx_int)

            output_pcl = output_pcl.transform(np.linalg.inv(mat1) @ mat2)
            output_pcl = check_occlusion_no_off(input_depth / depth_div, output_pcl, camera_matrix)
            input_pcl = input_pcl.transform(mat1)
            output_pcl = output_pcl.transform(mat1)
            target_pcl = target_pcl.transform(mat2)

            input_pcl.points = open3d.utility.Vector3dVector(
                [point for point in np.asarray(input_pcl.points) if point[2] > 0.05])
            output_pcl.points = open3d.utility.Vector3dVector(
                [point for point in np.asarray(output_pcl.points) if point[2] > 0.05])
            target_pcl.points = open3d.utility.Vector3dVector(
                [point for point in np.asarray(target_pcl.points) if point[2] > 0.05])
            output_pcl = output_pcl.paint_uniform_color(np.array([0.0, 1.0, 0.0]))

            visualiser([input_pcl], mat2, 'input')
            visualiser([input_pcl, target_pcl], mat2, 'gt')
            visualiser([input_pcl.paint_uniform_color(np.array([0.0, 1.0, 0.0])), output_pcl.paint_uniform_color(np.array([1.0, 0.0, 0.0]))], mat2, 'predicted')

    save_visu = True
    visu_path = 'reconstructions/0_5/'
    if save_visu:
        fig, ax = plt.subplots(nrows=num_samples, ncols=3, figsize=(2, 2), dpi=80)
        title = f'Dataset Furniture - Test - {start_idx} - {start_idx + num_samples - 1}'
        col_names = ['Input only', 'Ground truth', 'Predicted']
        fig.suptitle(title, fontsize=16)
        for i, a in enumerate(ax[0]):
            a.set_title(col_names[i])

        for i, (input_idx, target_idx) in enumerate(range[:num_samples]):
            print(os.path.join(visu_path, f'{input_idx}_input.png'))
            input_img = cv.imread(os.path.join(visu_path, f'{input_idx}_input.png'), flags=cv.IMREAD_COLOR)
            target_img = cv.imread(os.path.join(visu_path, f'{input_idx}_gt.png'), flags=cv.IMREAD_COLOR)
            output_img = cv.imread(os.path.join(visu_path, f'{input_idx}_predicted.png'), flags=cv.IMREAD_COLOR)
            ax[i][0].imshow(input_img.astype(np.uint8))
            ax[i][1].imshow(target_img.astype(np.uint8))
            ax[i][2].imshow(output_img.astype(np.uint8))
        plt.show()
