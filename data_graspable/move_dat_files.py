import os
from utils import *
from shutil import copy

all_files = listdir_full_path('/home/witsemp/Uczelnia/Magisterka/Data/dataset_3categories_10instances')
dat_files = [file for file in all_files if os.path.splitext(file)[1] == '.dat']
dest_path = '/home/witsemp/Uczelnia/Magisterka/Data/Dataset_DB_3C_10IN/locs'
for file in dat_files:
    copy(file, dest_path)