import os
import numpy
from data_graspable.utils import *
import natsort


def max_dist_value(path: str):
    _, _, _, _, dists = get_rgbdpd_paths(path)
    return np.max([np.max(cv.imread(dist_image_path, cv.IMREAD_ANYDEPTH)) for dist_image_path in dists])

def max_depth_value(path: str):
    depths = natsort.natsorted(
        [file for file in listdir_full_path(path) if os.path.splitext(os.path.basename(file))[1] == '.png'])
    return np.max([np.max(cv.imread(dist_image_path, cv.IMREAD_ANYDEPTH)) for dist_image_path in depths])

if __name__ == '__main__':
    dist_path = '/home/witsemp/Uczelnia/Magisterka/RealDataTest/01845'
    val = max_depth_value(dist_path)
    print(val)