import os
from PIL import Image
import cv2
from datasets import readpfm as rp
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from datasets.data_io import read_all_lines, pfm_imread
import random
from utils.config import *

def mb_loader(filepath):
    train_path = sys_root + filepath
    test_path = sys_root + filepath
    gt_path = sys_root + filepath

    train_left = []
    train_right = []
    train_gt = []

    for c in os.listdir(train_path):
        train_left.append(os.path.join(train_path, c, 'im0.png'))
        train_right.append(os.path.join(train_path, c, 'im1.png'))
        train_gt.append(os.path.join(gt_path, c, 'disp0.pfm'))

    test_left = []
    test_right = []
    for c in os.listdir(test_path):
        test_left.append(os.path.join(test_path, c, 'im0.png'))
        test_right.append(os.path.join(test_path, c, 'im1.png'))

    train_left = sorted(train_left)
    train_right = sorted(train_right)
    train_gt = sorted(train_gt)
    test_left = sorted(test_left)
    test_right = sorted(test_right)

    return train_left, train_right, train_gt, test_left, test_right

class MyDataset(data.Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.training = training
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

        self.img_transorm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def load_disp(self, filename):
        filename = os.path.normpath(filename)
        data, scale = pfm_imread(filename)
        data = np.nan_to_num(data, nan=0.0)
        data[np.isinf(data)] = 0.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def horizontal_flip(self, img):
        img_np = np.array(img)
        img_np = np.flip(img_np, axis=1)
        img = Image.fromarray(img_np)
        return img

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
            disparity = Image.fromarray(np.ascontiguousarray(disparity, dtype=np.float32))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size

            # random resize
            s = np.random.uniform(0.95, 1.05, 1)
            rw, rh = int(np.round(w*s)), int(np.round(h*s))
            left_img = left_img.resize((rw, rh), Image.NEAREST)
            right_img = right_img.resize((rw, rh), Image.NEAREST)
            disparity = disparity.resize((rw, rh), Image.NEAREST)
            disparity = Image.fromarray(np.array(disparity) * s)

            # random horizontal flip
            p = np.random.rand(1)
            if p >= 0.5:
                left_img = self.horizontal_flip(left_img)
                right_img = self.horizontal_flip(right_img)
                disparity = self.horizontal_flip(disparity)

            w, h = left_img.size
            tw, th = 640, 320
            # tw, th = (w // 4 // 32) * 32, (h // 4 // 32) * 32
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1+tw, y1+th))
            right_img = right_img.crop((x1, y1, x1+tw, y1+th))
            disparity = disparity.crop((x1, y1, x1+tw, y1+th))

            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            disparity = np.array(disparity)
            disparity_low = cv2.resize(disparity, (tw//4, th//4), interpolation=cv2.INTER_NEAREST)

            # return left_img, right_img, disparity
            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_low": disparity_low}

        else:
            w, h = left_img.size

            # normalize
            left_img = self.img_transorm(left_img)
            right_img = self.img_transorm(right_img)

            # pad
            top_pad = -(-h // 32) * 32 - h
            right_pad = -(-w // 32) * 32 - w

            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            # pad disparity gt
            if disparity is not None:
                disparity = np.array(disparity)
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            # return left_img, right_img, disparity
            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]
                        }
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}

if __name__ == '__main__':
    train_left, train_right, train_gt, _, _ = mb_loader(sys_root+"/MiddEval3-data-Q/", res='Q')
    H, W = 0, 0
    for l in train_right:
        left_img = Image.open(l).convert('RGB')
        h, w = left_img.size
        H += h
        W += w
    print(H / 15, W / 15)
