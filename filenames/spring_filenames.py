import os
import random

random_nums = random.sample(range(5000), 100)

files_l, files_r, files_d = [], [], []

root = './data/Spring'
path = 'train'
for folder in [os.path.join(path, folder, 'frame_left') for folder in sorted(os.listdir(os.path.join(root, path)))]:
    files_l = files_l + [os.path.join(folder, img) for img in sorted(os.listdir(os.path.join(root, folder)))]
for folder in [os.path.join(path, folder, 'frame_right') for folder in sorted(os.listdir(os.path.join(root, path)))]:
    files_r = files_r + [os.path.join(folder, img) for img in sorted(os.listdir(os.path.join(root, folder)))]
for folder in [os.path.join(path, folder, 'disp1_left') for folder in sorted(os.listdir(os.path.join(root, path)))]:
    files_d = files_d + [os.path.join(folder, disp) for disp in sorted(os.listdir(os.path.join(root, folder)))]

spring_f = open('spring_train.txt', 'w')
spring_val_f = open('spring_val.txt', 'w')
for index, (img_l, img_r, disp) in enumerate(zip(files_l, files_r, files_d)):
    line = img_l + ' ' + img_r + ' ' + disp
    if index in random_nums:
        spring_val_f.write(line + '\n')
    else:
        spring_f.write(line + '\n')

spring_f.close()
spring_val_f.close()
