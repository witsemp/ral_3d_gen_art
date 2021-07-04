import cv2 as cv
import numpy as np
import torch
import open3d
from lxml import etree
import re
import os


def pcl_from_images(depth_image, camera_matrix, return_o3d=True):
    pcl = []
    if len(np.shape(depth_image)) == 2:
        rows, cols = np.shape(depth_image)
    else:
        rows, cols, _ = np.shape(depth_image)
    for row in range(rows):
        for col in range(cols):
            world_coords = depth_image[row, col] * np.dot(camera_matrix, np.array([col, row, 1]).reshape(3, 1))
            if world_coords[2] < far:
                pcl.append(world_coords)
    pcl = np.array(pcl).reshape(len(pcl), 3)
    if return_o3d:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcl)
        return pcd
    else:
        return pcl

def manual_resize(image, dst_shape):
    h_in, w_in = image.shape
    h_dst, w_dst = dst_shape
    h_range = list(range(h_in))
    w_range = list(range(w_in))
    h_stride = h_in // h_dst
    w_stride = w_in // w_dst
    dst_image = np.zeros(dst_shape)
    for i, row in enumerate(h_range[0::h_stride]):
        for j, col in enumerate(w_range[0::w_stride]):
            dst_image[i, j] = image[row, col]
    return dst_image.astype(np.float32)

def image_from_pcl(pcl, img_shape, camera_matrix):
    depth_image = np.zeros(img_shape).astype(np.float32)
    h, w = img_shape
    for point in pcl.points:
        d = point[2]
        world_coords = point.reshape(3, 1) / d
        # print(world_coords)
        u, v, _ = np.linalg.inv(camera_matrix) @ world_coords
        try:
            u, v = int(np.ceil(u)), int(np.ceil(v))
        except:
            pass
        if 0 < u < w and 0 < v < h and d > 0:
            if (depth_image[v, u] == 0) or (depth_image[v, u] > d):
                depth_image[v, u] = d
    return depth_image


def image_from_pcl_w_offset(pcl, img_shape, offset, camera_matrix):
    depth_image = np.zeros(img_shape).astype(np.uint16)
    h, w = img_shape
    u_off, v_off = offset
    for point in pcl.points:
        d = point[2]
        world_coords = point.reshape(3, 1) / d
        u, v, _ = np.linalg.inv(camera_matrix) @ world_coords
        u, v = int(np.ceil(u) - u_off), int(np.ceil(v) - v_off)
        if 0 < u < w and 0 < v < h and d > 0:
            if (depth_image[v, u] == 0) or depth_image[v, u] > d:
                # print('here')
                depth_image[v, u] = d
    return depth_image


def pcl_from_images_with_offset(depth_image, camera_matrix, offset, far, return_o3d=True):
    pcl = []
    u_off, v_off = offset
    if len(np.shape(depth_image)) == 2:
        rows, cols = np.shape(depth_image)
    else:
        rows, cols, _ = np.shape(depth_image)
    for row in range(rows):
        for col in range(cols):
            world_coords = depth_image[row, col] * np.dot(camera_matrix,
                                                          np.array([col + u_off, row + v_off, 1]).reshape(3, 1))
            if world_coords[2] < far:
                pcl.append(world_coords)
    pcl = np.array(pcl).reshape(len(pcl), 3)
    if return_o3d:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcl)
        return pcd
    else:
        return pcl


def pcl_from_rgbd(depth_image, rgb_image, camera_matrix):
    depth_o3d = open3d.geometry.Image(depth_image.astype(np.float32))
    rgb_o3d = open3d.geometry.Image(rgb_image)
    rgbd_o3d = open3d.geometry.RGBDImage()
    rgbd_o3d.depth = depth_o3d
    rgbd_o3d.color = rgb_o3d
    intrinsics = open3d.camera.PinholeCameraIntrinsic()
    intrinsics.intrinsic_matrix = camera_matrix
    pcl = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, intrinsics)
    return pcl


def get_transform_matrix(xml_file_path: str, index: int):
    rows = []
    root = etree.parse(xml_file_path)
    file = root.findall('File[@Index=' + '"' + str(index) + '"' + ']')
    for row_index in range(4):
        row = str(etree.tostring(file[0].findall('MatrixRow' + str(row_index))[0]))
        try:
            row = re.search(rf'<MatrixRow{row_index}>(.+?)</MatrixRow{row_index}>', row).group(1)
        except AttributeError:
            row = ''
        row = row.replace("(", '').replace(")", '').split(', ')
        row = [float(element) for element in row]
        rows.append(row)
    transform_matrix = np.array([rows[0], rows[1], rows[2], rows[3]])
    return transform_matrix


def voxelize(pcd: open3d.geometry.PointCloud, voxel_size: float = 0.05):
    return open3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)


def iou(predicted_grid, target_grid):
    pass


def check_occlusion(input_image: np.ndarray, pcl: open3d.geometry.PointCloud, camera_matrix: np.ndarray, offset):
    h, w = input_image.shape
    u_off, v_off = offset
    new_pcl = open3d.geometry.PointCloud()
    valid_points = []
    for point in pcl.points:
        d = point[2]
        world_coords = point.reshape(3, 1) / d
        u, v, _ = np.linalg.inv(camera_matrix) @ world_coords
        u, v = int(np.ceil(u) - u_off), int(np.ceil(v) - v_off)
        if 0 < u < w and 0 < v < h and d > 0:
            if d > input_image[v, u]:
                valid_points.append(point)
    valid_points = np.asarray(valid_points).reshape((len(valid_points), 3))
    new_pcl.points = open3d.utility.Vector3dVector(valid_points)
    return new_pcl


def check_occlusion_no_off(input_image: np.ndarray, pcl: open3d.geometry.PointCloud, camera_matrix: np.ndarray):
    h, w = input_image.shape
    new_pcl = open3d.geometry.PointCloud()
    valid_points = []
    for point in pcl.points:
        d = point[2]
        world_coords = point.reshape(3, 1) / d
        u, v, _ = np.linalg.inv(camera_matrix) @ world_coords
        u, v = int(np.ceil(u)), int(np.ceil(v))
        if 0 < u < w and 0 < v < h and d > 0:
            if d > input_image[v, u]:
                # print(input_image[v, u])
                valid_points.append(point)
    valid_points = np.asarray(valid_points).reshape((len(valid_points), 3))
    new_pcl.points = open3d.utility.Vector3dVector(valid_points)
    return new_pcl


f_depth_x = 525.0
f_depth_y = 525.0
c_depth_x = 320
c_depth_y = 240.5
camera_matrix = np.array([[1 / f_depth_x, 0., -c_depth_x / f_depth_x],
                          [0., 1 / f_depth_y, -c_depth_y / f_depth_y],
                          [0., 0., 1.]])
camera_matrix_opencv = np.array(
    [[f_depth_x, 0., c_depth_x],
     [0., f_depth_y, c_depth_y],
     [0., 0., 1.]])

far = 10
depth_div_unity = 1.09
rgb_div = 255.0
scale_unity = far / depth_div_unity
scale_db = 30562
