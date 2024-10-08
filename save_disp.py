from __future__ import print_function, division
import argparse
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import time
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
from datasets.flow_IO import writeDispFile
from utils.config import *
# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='A Lightweight Target-Driven Network of Stereo Matching for Inland Waterways (LTNet)')
parser.add_argument('--model', default='LTNet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='usvinland', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath_usv', default=sys_root+"/USVInland/Stereo Matching/Low_Res_640_320/", help='usvinland data path')
parser.add_argument('--datapath_12', default=sys_root+"/KITTI/KITTI2012/data_stereo_flow/", help='kitti 12 data path')
parser.add_argument('--datapath_15', default=sys_root+"/KITTI/KITTI2015/data_scene_flow/", help='kitti 15 data path')
parser.add_argument('--datapath_mb', default=sys_root+"/Middlebury/", help='middlebury data path')
parser.add_argument('--datapath_eth', default=sys_root+"/ETH3D/", help='eth3d data path')
parser.add_argument('--datapath_spring', default=sys_root+"/Spring/", help='spring data path')
parser.add_argument('--datapath_sceneflow', default=sys_root+"/SceneFlow/", help='sceneflow data path')
parser.add_argument('--testlist',default='./filenames/usvinland_val.txt', help='testing list')
parser.add_argument('--kfold', type=int, default=0, help='Kfold num', choices=range(6))

parser.add_argument('--num_workers', type=int, default=0, help='num workers')

parser.add_argument('--loadckpt', default='./checkpoints/sceneflow/second/best.ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--submit', action='store_true', help='submit the result to spring')
parser.add_argument('--pred_right', action='store_true', help='submit the result to spring')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
if args.dataset == 'usvinland':
    if args.kfold != 0:
        print('Kfold:', args.kfold)
        args.testlist = args.testlist.replace('.txt', '_' + str(args.kfold) + '.txt')
    test_dataset = StereoDataset(args.datapath_usv, args.testlist, False, False)
elif args.dataset == 'kitti':
    test_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.testlist, False)
elif args.dataset == 'middlebury':
    test_dataset = StereoDataset(args.datapath_mb, args.testlist, False)
elif args.dataset == 'eth3d':
    test_dataset = StereoDataset(args.datapath_eth, args.testlist, False)
elif args.dataset == 'spring':
    test_dataset = StereoDataset(args.datapath_spring, args.testlist, False, args.pred_right)
elif args.dataset =='sceneflow':
    test_dataset = StereoDataset(args.datapath_sceneflow, args.testlist, False)

TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.num_workers, drop_last=False)
print('dataset', args.dataset)
print('testlist:', args.testlist)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

###load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

save_dir = './out/' + args.dataset
if args.dataset == 'usvinland' and args.kfold != 0: save_dir = os.path.join(save_dir, 'Kfold_' + str(args.kfold))

def test():
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'pseudo'), exist_ok=True)


    # x, y = torch.randn(1, 3, 640, 640).cuda(), torch.randn(1, 3, 640, 640).cuda()
    # for i in range(50):
    #     output = model(x, y)

    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        torch.cuda.synchronize()
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))

        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            assert len(disp_est.shape) == 2

            if right_pad == 0: disp_est = np.array(disp_est[top_pad:, :], dtype=np.float32)
            else: disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)

            # submission
            if 'spring' in args.dataset: fn = fn.replace('frame', 'disp1')

            filename = fn.replace('jpg', 'png')
            os.makedirs(os.path.join(save_dir, filename[:filename.rfind("/")]), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'pseudo', filename[:filename.rfind("/")]), exist_ok=True)

            fn = os.path.join(save_dir, filename)
            pfn = os.path.join(save_dir, 'pseudo', filename)
            if args.pred_right: disp_est = cv2.flip(disp_est, 1)


            if 'usvinland' in args.dataset: disp_est_uint = np.round(disp_est / 50. * 255.).astype(np.uint8)
            else: disp_est_uint = np.round(disp_est * 256.).astype(np.uint16)


            if 'usvinland' in args.dataset: disp_pseudo = cv2.applyColorMap(cv2.convertScaleAbs(np.round(disp_est / 50. * 255.).astype(np.uint8), alpha=5), cv2.COLORMAP_PLASMA) # cv2.COLORMAP_JET
            elif 'kitti' in args.dataset: disp_pseudo = cv2.applyColorMap(cv2.convertScaleAbs(disp_est.astype(np.uint16), alpha=5), cv2.COLORMAP_PLASMA)
            elif 'middlebury' in args.dataset: disp_pseudo = cv2.applyColorMap(cv2.convertScaleAbs(disp_est.astype(np.uint16), alpha=2), cv2.COLORMAP_PLASMA)
            elif 'eth3d' in args.dataset: disp_pseudo = cv2.applyColorMap(cv2.convertScaleAbs(np.round(disp_est / args.maxdisp * 255.).astype(np.uint8), alpha=2.6), cv2.COLORMAP_PLASMA)
            else: disp_pseudo = cv2.applyColorMap(cv2.convertScaleAbs(np.round(disp_est / args.maxdisp * 255.).astype(np.uint16), alpha=5), cv2.COLORMAP_PLASMA)

            if 'spring' in args.dataset and args.submit: writeDispFile(disp_est, fn.replace('png', 'dsp5'))
            else: cv2.imwrite(fn, disp_est_uint)
            cv2.imwrite(pfn, disp_pseudo)

# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    if args.model == 'LTNet':
        disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]

if __name__ == '__main__':
    test()
