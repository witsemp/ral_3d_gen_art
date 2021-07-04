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


def most_frequent(l):
    return max(set(l), key=l.count)


batch_size = 1
max_depth_value = 39999
depth_div = 5000
visu_dataset_path = '/home/witsemp/Uczelnia/Magisterka/Data/Dataset_DB_Furniture_Test/train'

device = torch.device('cpu')
model_path = '../models/UNet_DatasetDB_Furniture.pth'
model = torch.load(model_path, map_location=device)
model.eval()

visu_dataset = FurnitureDataset(dataset_path=visu_dataset_path, depth_div=max_depth_value, transform=None)
visu_dataloader = DataLoader(dataset=visu_dataset, batch_size=batch_size, shuffle=True)
input_tensor, target_tensor, mat1, mat2 = next(iter(visu_dataloader))
output_depth = model(input_tensor)
output_depth = output_depth[0].detach().cpu()
output_depth = image_from_tensor(output_depth, max_depth_value)
input_tensor, target_depth, mat1, mat2 = input_tensor[0], target_tensor[0], mat1[0].numpy(), mat2[0].numpy()
input_image = image_from_tensor(input_tensor, scale=False)
input_rgb, input_depth, projected_depth, target_depth = input_image[..., :3] * 255.0, \
                                                        input_image[..., 3] * max_depth_value, \
                                                        input_image[..., 4] * max_depth_value, \
                                                        image_from_tensor(target_depth, max_depth_value)


input_rgb = cv.resize(input_rgb, (640, 480))
input_depth = cv.resize(input_depth, (640, 480), interpolation=cv.INTER_AREA)
input_depth = np.asarray(input_depth, dtype=np.float32) / depth_div
output_depth = cv.resize(output_depth, (640, 480), interpolation=cv.INTER_AREA)
# cv.imshow('test', output_depth)
# cv.waitKey(0)
output_depth = np.asarray(output_depth, dtype=np.float32) / depth_div



input_pcl = pcl_from_rgbd(input_depth, input_rgb.astype(np.uint8), camera_matrix_opencv)
output_pcl = pcl_from_images(output_depth, camera_matrix)
output_pcl = output_pcl.paint_uniform_color(np.array([1.0, 0.0, 0.0]))

# transform output point cloud to input camera position
output_pcl = output_pcl.transform(np.linalg.inv(mat1) @ mat2)
# check for occlusions
output_pcl = check_occlusion_no_off(input_depth / depth_div, output_pcl, camera_matrix)
# transform point clouds to common frame

input_pcl = input_pcl.transform(mat1)
output_pcl = output_pcl.transform(mat1)
output_pcl = output_pcl.paint_uniform_color(color=np.array([0.0, 1.0, 0.0]).astype(float))


input_pcl = np.asarray(input_pcl.points)
output_pcl = np.asarray(output_pcl.points)
all_points = np.concatenate((input_pcl, output_pcl))
points_no_floor = [point for point in all_points if point[2] > 0.05]

pcl = open3d.geometry.PointCloud()
pcl.points = open3d.utility.Vector3dVector(points_no_floor)

open3d.visualization.draw_geometries([pcl])
