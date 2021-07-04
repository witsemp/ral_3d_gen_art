import natsort
import open3d.visualization
from point_cloud_utils.point_cloud import *
from scipy.spatial.transform import Rotation
import os
import cv2 as cv
import numpy as np
from dataset_db_furniture.visualise_results import visualiser_static
from dataset_db.utils import listdir_full_path


def get_second_cam_pose(cam_pose: np.ndarray, d: float, angle: float) -> np.ndarray:
    # Forward
    mat_forward = np.identity(4)
    mat_forward[2, 3] = d
    new_cam_pose = cam_pose @ mat_forward

    # Make orientation horizontal
    rot_rpy = Rotation.from_euler('x', angle, degrees=True)
    rot_local = np.identity(4)
    rot_local[:-1, :-1] = rot_rpy.as_matrix()
    new_cam_pose = new_cam_pose @ rot_local

    # Rotate 180 degrees in local Y
    rot_rpy2 = Rotation.from_euler('y', 180, degrees=True)
    rot_local2 = np.identity(4)
    rot_local2[:-1, :-1] = rot_rpy2.as_matrix()
    new_cam_pose = new_cam_pose @ rot_local2

    # Go back to original orientation
    new_cam_pose = new_cam_pose @ np.linalg.inv(rot_local)

    # Translate backwards
    mat_backward = np.identity(4)
    mat_backward[2, 3] = -d
    new_cam_pose = new_cam_pose @ mat_backward
    return new_cam_pose


data_path = '/home/witsemp/Uczelnia/Magisterka/RealDataTest/01845'
rgbs = natsort.natsorted(
    [file for file in listdir_full_path(data_path) if os.path.splitext(os.path.basename(file))[1] == '.jpg'])
depths = natsort.natsorted(
    [file for file in listdir_full_path(data_path) if os.path.splitext(os.path.basename(file))[1] == '.png'])
depth_div = 1000


npy_path = 'projected'
os.makedirs(npy_path, exist_ok=True)
write = True
if write:
    for (rgb_path, depth_path) in list(zip(rgbs, depths)):
        rgb_img = cv.cvtColor(cv.imread(rgb_path, flags=cv.IMREAD_COLOR), code=cv.COLOR_BGR2RGB)
        depth_img = cv.imread(depth_path, flags=cv.IMREAD_ANYDEPTH).astype(np.float32)
        pcl = pcl_from_images(depth_img / depth_div, camera_matrix)
        second_cam_pose = get_second_cam_pose(np.identity(4), 1.3, angle=55)
        mesh_frame_init = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])
        mesh_frame_mirror = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])
        mesh_frame_mirror = mesh_frame_mirror.transform(second_cam_pose)
        visualiser_static([pcl, mesh_frame_init, mesh_frame_mirror], np.identity(4))


        pcl = pcl.transform(np.linalg.inv(second_cam_pose))
        img_second_pose = image_from_pcl(pcl, (480, 640), camera_matrix) * depth_div
        print(np.unique(img_second_pose))

        file_name = f'{os.path.join(npy_path, os.path.splitext(os.path.basename(depth_path))[0])}.png'
        cv.imwrite(file_name, img_second_pose.astype(np.uint16))

        img_test = cv.imread(file_name, flags=cv.IMREAD_ANYDEPTH)
        print(np.unique(img_test))