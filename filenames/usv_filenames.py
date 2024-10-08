import os
import random

import numpy as np

path = './data/USVInland/Stereo Matching/Low_Res_640_320'
files_l = sorted(os.listdir(os.path.join(path, 'Left_Img_Rectified')))
files_l = ['Left_Img_Rectified/' + img for img in files_l]
files_r = sorted(os.listdir(os.path.join(path, 'Right_Img_Rectified')))
files_r = ['Right_Img_Rectified/' + img for img in files_r]
files_d = sorted(os.listdir(os.path.join(path, 'Disp_Map')))
files_d = ['Disp_Map/' + img for img in files_d]

k = 5
index_list = list(range(180))

def rand_fold(data, k):
    val_index = []
    fold_num = len(data) // k

    for _ in range(k):
        sample = random.sample(data, fold_num)
        val_index.append(sample)
        data = [i for i in data if i not in sample]
    return val_index

imgs_ranges = rand_fold(index_list, k)
np.save('./filenames/USVInland_imgs_ranges.npy', imgs_ranges)


for i in range(k):
    usv_f = open('./filenames/usvinland_train_' + str(i+1) + '.txt', 'w')
    usv_val_f = open('./filenames/usvinland_val_' + str(i+1) + '.txt', 'w')
    for j, (img_l, img_r, disp) in enumerate(zip(files_l, files_r, files_d)):
        line = img_l + ' ' + img_r + ' ' + disp
        if j in imgs_ranges[i]: usv_val_f.write(line + '\n')
        else: usv_f.write(line + '\n')

    usv_f.close()
    usv_val_f.close()
