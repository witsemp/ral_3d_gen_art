import os.path
import cv2 as cv
import natsort
import numpy as np
import open3d.visualization
import torch

from point_cloud_utils.point_cloud import *
from data_graspable.utils import get_rgbdp_paths, listdir_full_path
from data_graspable.data import *
from matplotlib import pyplot as plt

batch_size = 1
max_depth_value = 30562

visu_dataset_path = '/home/witsemp/Uczelnia/Magisterka/Data/VisualisationDataset'
model_path = '../models/UNet_DatasetDB_3C_10IN_SSIM.pth'

device = torch.device('cpu')
model = torch.load(model_path, map_location=device)
model.eval()

visu_dataset = VisualisationDataset(dataset_path=visu_dataset_path, depth_div=max_depth_value, transform=None)
visu_dataloader = DataLoader(dataset=visu_dataset, batch_size=batch_size, shuffle=True)

input_tensor, target_tensor, scene_rgb, scene_depth, target_scene_rgb, target_scene_depth, input_loc, target_loc, mat1, mat2 = next(
    iter(visu_dataloader))

output_depth = model(input_tensor)
output_depth = output_depth[0].detach().cpu()
output_depth = image_from_tensor(output_depth, max_depth_value)

input_tensor, target_depth, scene_rgb, scene_depth, target_scene_rgb, target_scene_depth, input_loc, target_loc, mat1, mat2 = \
    input_tensor[0], \
    target_tensor[0], \
    scene_rgb[0].numpy(), \
    scene_depth[0].numpy(), \
    target_scene_rgb[0].numpy(), \
    target_scene_depth[0].numpy(), \
    input_loc[0].numpy(), \
    target_loc[0].numpy(), \
    mat1[0].numpy(), \
    mat2[0].numpy()

input_image = image_from_tensor(input_tensor, scale=False)
input_rgb, input_depth, projected_depth, target_depth = input_image[..., :3] * 255.0, \
                                                        input_image[..., 3] * max_depth_value, \
                                                        input_image[..., 4] * max_depth_value, \
                                                        image_from_tensor(target_depth, max_depth_value)
cv.imwrite('samples/rgb.png', input_rgb)
cv.imwrite('samples/in_d.png', input_depth.astype(np.uint16))
cv.imwrite('samples/proj.png', projected_depth.astype(np.uint16))
cv.imwrite('samples/output.png', output_depth.astype(np.uint16))

depth_div = 5000

h, w, _ = input_rgb.shape
u_input, v_input = input_loc
input_loc = (int(u_input - w // 2), int(v_input - h // 2))

u_target, v_target = target_loc
target_loc = (int(u_target - w // 2), int(v_target - h // 2))

# compute point clouds
input_scene_pcl = pcl_from_rgbd(scene_depth / depth_div, scene_rgb, camera_matrix_opencv)
output_pcl = pcl_from_images_with_offset(output_depth / depth_div, camera_matrix, offset=target_loc, far=30000,
                                         return_o3d=True)
output_pcl_occl = open3d.geometry.PointCloud()
output_pcl_occl.points = output_pcl.points

# open3d.visualization.draw_geometries([input_scene_pcl])

# without filtering
input_scene_pcl = input_scene_pcl.transform(mat1)
output_pcl = output_pcl.transform(mat2)
output_pcl = output_pcl.paint_uniform_color(color=np.array([0.0, 1.0, 0.0]).astype(float))
# open3d.visualization.draw_geometries([input_scene_pcl, output_pcl])


# transform output point cloud to input camera position
output_pcl_occl = output_pcl_occl.transform(np.linalg.inv(mat1) @ mat2)
# check for occlusions
output_pcl_occl = check_occlusion(input_depth / depth_div, output_pcl_occl, camera_matrix, input_loc)
# transform point clouds to common frame
output_pcl_occl = output_pcl_occl.transform(mat1)

output_pcl_occl = output_pcl_occl.paint_uniform_color(color=np.array([0.0, 1.0, 0.0]).astype(float))
open3d.visualization.draw_geometries([input_scene_pcl, output_pcl_occl])

