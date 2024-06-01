# -*- coding: UTF-8 -*-
"""
This script samples randomly data from a directory by given arg 'number_to_sample_of_each_directory'
"""
import sys
import os
import random, shutil
from tqdm import tqdm

def safe_makedirs(p):
    """
    Make directory with existence check
    :param p: target directory path
    :return: NA
    """
    if os.path.exists(p):
        inp = input("Warning: The directory '%s' already exists, "
                    "it will be removed and new created."
                    "\nType N or n "
                    "to terminate the program OR any key to continue: " % p)
        if inp == 'N' or inp == 'n':
            sys.exit(2)
        print('Removing directory', p)
        shutil.rmtree(p)
        print('Removed', p)
    try:
        os.makedirs(p)
    except OSError as e:
        if e.errno != 17:
            raise Exception('Failed to create directory', p)

src_dir = './Oulu'#被分割的文件路径
tar_dir = './Oulu_split'#被分割的数据集存储的文件路径
val_ratio = 0.1
safe_makedirs(tar_dir)
#tar_dir_train = os.path.join(tar_dir, 'train')
tar_dir_val = os.path.join(tar_dir, 'val')
#os.makedirs(tar_dir_train)
os.makedirs(tar_dir_val)
sub_dirs = os.listdir(src_dir)
for sub_dir in sub_dirs:
    #os.mkdir(os.path.join(tar_dir_train, sub_dir))
    os.mkdir(os.path.join(tar_dir_val, sub_dir))
for sub_dir in sub_dirs:
    dir_path = os.path.join(src_dir, sub_dir)
    image_names = os.listdir(dir_path)
    random.shuffle(image_names)
    num_images = len(image_names)
    val_num = int(num_images * val_ratio)
    print('Processing', sub_dir)
    for cnt, image_name in enumerate(tqdm(image_names)):
        if cnt < val_num:
            shutil.copy(os.path.join(dir_path, image_name), os.path.join(tar_dir_val, sub_dir))
        else:
            shutil.copy(os.path.join(dir_path, image_name), os.path.join(tar_dir_train, sub_dir))

print('All done')