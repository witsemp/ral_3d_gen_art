import numpy as np

from point_cloud_utils.point_cloud import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import plotly.graph_objects as go


def plotFromVoxels(voxels):
    if len(voxels.shape) > 3:
        x_d = voxels.shape[0]
        y_d = voxels.shape[1]
        z_d = voxels.shape[2]
        v = voxels[:, :, :, 0]
        v = np.reshape(v, (x_d, y_d, z_d))
    else:
        v = voxels
    x, y, z = v.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='z', c='red')
    plt.show()


def plotlyFromVoxels(voxels, marker_size=5, title=''):
    if len(voxels.shape) > 3:
        x_d = voxels.shape[0]
        y_d = voxels.shape[1]
        z_d = voxels.shape[2]
        v = voxels[:, :, :, 0]
        v = np.reshape(v, (x_d, y_d, z_d))
    else:
        v = voxels
    x, y, z = v.nonzero()
    marker_data = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        marker=go.scatter3d.Marker(size=marker_size, symbol='square'),
        opacity=0.8,
        mode='markers'
    )
    fig = go.Figure(data=marker_data)
    fig.show()


def voxelization(pc_25d, save=True, save_name='test.npz'):
    vox_res = 256

    x_max = max(pc_25d[:, 0])
    x_min = min(pc_25d[:, 0])
    y_max = max(pc_25d[:, 1])
    y_min = min(pc_25d[:, 1])
    z_max = max(pc_25d[:, 2])
    z_min = min(pc_25d[:, 2])
    step = round(max([x_max - x_min, y_max - y_min, z_max - z_min]) / (vox_res - 1), 4)
    x_d_s = int((x_max - x_min) / step)
    y_d_s = int((y_max - y_min) / step)
    z_d_s = int((z_max - z_min) / step)

    vox = np.zeros((x_d_s + 1, y_d_s + 1, z_d_s + 1, 1), dtype=np.int8)
    for k, p in enumerate(pc_25d):
        if k % 50000 == 0:
            print(k)
        ##### voxlization 25d
        xd = int((p[0] - x_min) / step)
        yd = int((p[1] - y_min) / step)
        zd = int((p[2] - z_min) / step)
        if xd >= vox_res or yd >= vox_res or zd >= vox_res:
            # print("xd>=vox_res or yd>=vox_res or zd>=vox_res")
            continue
        if xd > x_d_s or yd > y_d_s or zd > z_d_s:
            # print("xd>x_d_s or yd>y_d_s or zd>z_d_s")
            continue

        vox[xd, yd, zd, 0] = 1
    if save:
        np.savez_compressed(save_name, vox)

    return vox


def voxelization_fixed_res(pc_25d, save=True, save_name='test.npz'):
    vox_res = 256
    x_max = max(pc_25d[:, 0])
    x_min = min(pc_25d[:, 0])
    y_max = max(pc_25d[:, 1])
    y_min = min(pc_25d[:, 1])
    z_max = max(pc_25d[:, 2])
    z_min = min(pc_25d[:, 2])
    step = round(max([x_max - x_min, y_max - y_min, z_max - z_min]) / (vox_res - 1), 4)
    x_d_s = int((x_max - x_min) / step)
    y_d_s = int((y_max - y_min) / step)
    z_d_s = int((z_max - z_min) / step)

    vox = np.zeros((vox_res, vox_res, vox_res), dtype=np.int8)
    for k, p in enumerate(pc_25d):
        if k % 50000 == 0:
            print(k)
        ##### voxlization 25d
        xd = int((p[0] - x_min) / step)
        yd = int((p[1] - y_min) / step)
        zd = int((p[2] - z_min) / step)
        if xd >= vox_res or yd >= vox_res or zd >= vox_res:
            # print("xd>=vox_res or yd>=vox_res or zd>=vox_res")
            continue
        if xd > x_d_s or yd > y_d_s or zd > z_d_s:
            # print("xd>x_d_s or yd>y_d_s or zd>z_d_s")
            continue

        vox[xd, yd, zd] = 1
    if save:
        np.savez_compressed(save_name, vox)

    return vox


def single_depth_2_pc(in_depth_path, depth_scale=1 / 5000, return_03d=False):
    depth = cv.imread(in_depth_path, flags=cv.IMREAD_ANYDEPTH)
    depth = np.asarray(depth, dtype=np.float32) * depth_scale
    xyz_pc = pcl_from_images(depth, camera_matrix, return_o3d=return_03d)
    if return_03d:
        open3d.visualization.draw_geometries([xyz_pc])
    return xyz_pc
