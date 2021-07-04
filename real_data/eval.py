from data import RealSensorDataset, image_from_tensor
from torch.utils.data import DataLoader
import torch
import cv2 as cv
import os
import numpy as np

depth_div = 3399.0
dataset_path = 'real_data_npy'
save_path = 'predictions'
visu_dataset = RealSensorDataset(dataset_path=dataset_path, depth_div=depth_div, transform=None)
visu_dataloader = DataLoader(dataset=visu_dataset, batch_size=1, shuffle=True)
device = torch.device('cpu')
model_path = '../models/UNet_DatasetDB_Furniture.pth'
model = torch.load(model_path, map_location=device)
model.eval()

# def window_mean(img, idx):
#     window = img[id]

for i, input_tensor in enumerate(visu_dataloader):
    input_tensor = input_tensor.to(device)
    output_tensor = model(input_tensor)

    input_image = image_from_tensor(input_tensor[0].detach().cpu(), scale=False)
    output_depth = output_tensor[0].detach().cpu()
    input_rgb, input_depth, projected_depth = input_image[..., :3] * 255.0, input_image[..., 3] * depth_div, input_image[..., 4] * depth_div
    output_depth = image_from_tensor(output_depth, depth_div)
    output_depth[output_depth < 0] = 0
    cv.imwrite(os.path.join(save_path, f'{i}_rgb' + '.png'), input_rgb.astype(np.uint8))
    cv.imwrite(os.path.join(save_path, f'{i}_depth' + '.png'), input_depth.astype(np.uint16))
    cv.imwrite(os.path.join(save_path, f'{i}_projected' + '.png'), projected_depth.astype(np.uint16))
    cv.imwrite(os.path.join(save_path, f'{i}_output' + '.png'), output_depth.astype(np.uint16))
    test_img = cv.imread(os.path.join(save_path, f'{i}_output' + '.png'), flags=cv.IMREAD_ANYDEPTH)
    print(np.unique(test_img))