from matplotlib import pyplot as plt
import cv2 as cv
from typing import List
import os
import numpy as np


def make_grid1(path1: str, num_rows: int, col_names: List):
    fig, ax = plt.subplots(nrows=num_rows, ncols=4, figsize=(30, 30))
    # fig.suptitle(title, fontsize=16)
    for i, a in enumerate(ax[0]):
        a.set_title(col_names[i])
    for row in range(num_rows):
        input_rgb_img = cv.imread(os.path.join(path1, f'{row}_rgb.png'), flags=cv.IMREAD_COLOR)
        input_depth_img = cv.imread(os.path.join(path1, f'{row}_depth.png'), flags=cv.IMREAD_ANYDEPTH)
        projected_depth_img = cv.imread(os.path.join(path1, f'{row}_projected.png'), flags=cv.IMREAD_ANYDEPTH)
        output_img = cv.imread(os.path.join(path1, f'{row}_output.png'), flags=cv.IMREAD_ANYDEPTH)
        ax[row][0].imshow(input_rgb_img.astype(np.uint8))
        ax[row][1].imshow(input_depth_img.astype(float))
        ax[row][2].imshow(projected_depth_img.astype(float))
        ax[row][3].imshow(output_img.astype(float))
    plt.show()


path = 'predictions'
make_grid1(path, 6, ['RGB', 'Depth', 'Projected', 'Output'])
