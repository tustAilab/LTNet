from __future__ import print_function, division
import argparse
import os
import random
# from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import time
from torch.utils.tensorboard import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train, model_loss_test
from models.submodule import *
from utils import *
from torch.utils.data import DataLoader
import gc
from utils.config import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

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
parser.add_argument('--trainlist', default='./filenames/usvinland_train.txt', help='training list')
parser.add_argument('--testlist', default='./filenames/usvinland_val.txt', help='testing list')
parser.add_argument('--kfold', type=int, default=0, help='Kfold num', choices=range(6))
parser.add_argument('--distill', action='store_true', help='Knowledge Distillation')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--num_workers', type=int, default=0, help='num workers')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="300:10", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--saveckpt', default='./checkpoints/', help='the directory to save checkpoints')
parser.add_argument('--logdir', default='./logs/', help='the directory to save logs')
parser.add_argument('--loadckpt', default='./checkpoints/sceneflow/second/best.ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--pred_right', action='store_true', help='submit the result to spring')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True

os.makedirs(args.saveckpt, exist_ok=True)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
if args.dataset == 'usvinland':
    if args.kfold != 0:
        print('Kfold:', args.kfold)
        args.trainlist = args.trainlist.replace('.txt', '_' + str(args.kfold) + '.txt')
        args.testlist = args.testlist.replace('.txt', '_' + str(args.kfold) + '.txt')
    train_dataset = StereoDataset(args.datapath_usv, args.trainlist, True, args.distill)
    test_dataset = StereoDataset(args.datapath_usv, args.testlist, False, False)
elif args.dataset == 'kitti':
    train_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.trainlist, True)
    test_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.testlist, False)
elif args.dataset == 'middlebury':
    train_dataset = StereoDataset(args.datapath_mb, args.trainlist, True)
    test_dataset = StereoDataset(args.datapath_mb, args.testlist, False)
elif args.dataset == 'eth3d':
    train_dataset = StereoDataset(args.datapath_eth, args.trainlist, True)
    test_dataset = StereoDataset(args.datapath_eth, args.testlist, False)
elif args.dataset == 'spring':
    train_dataset = StereoDataset(args.datapath_spring, args.trainlist, True, args.pred_right)
    test_dataset = StereoDataset(args.datapath_spring, args.testlist, False, args.pred_right)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
print('trainlist:', args.trainlist)
print('testlist:', args.testlist)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
torch.set_anomaly_enabled(True)
print("Parameter Count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# if args.dataset == 'usvinland':

#     prepare_input = lambda _: {"left":  torch.FloatTensor(1, 3, 320, 640).to('cuda'), "right":  torch.FloatTensor(1, 3, 320, 640).to('cuda')}
#     macs, params = get_model_complexity_info(model.module, input_res=(3, 320, 640), input_constructor=prepare_input, print_per_layer_stat=False, verbose=False)
#     print(f'ptflops: {{ macs: {macs}, params: {params} }}')

# load parameters
start_epoch = 0
best_epoch = 0
best_error = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.saveckpt) if (fn.startswith("checkpoint_") and fn.endswith(".ckpt"))]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.saveckpt, all_saved_ckpts[-1])
    print("loading the lastest model in saveckpt: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
    best_path = os.path.join(args.saveckpt, 'best.ckpt')
    if os.path.exists(best_path):
        best_model = torch.load(os.path.join(args.saveckpt, 'best.ckpt'))
        best_epoch = best_model['epoch']
        best_error = best_model['avg_test_scalars']['EPE'][0]
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))

def train():
    bestepoch = 0 if best_epoch == 0 else best_epoch
    error = 100 if best_error == 0 else best_error
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):

            #     break
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss, time.time() - start_time))


        # if (epoch_idx + 1) % args.save_freq == 0:
        #     checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.saveckpt, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss, time.time() - start_time))

        global_step = len(TrainImgLoader) * (epoch_idx + 1)
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["EPE"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["EPE"][0]

            checkpoint_data = {'epoch': epoch_idx, 'step': global_step, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'avg_test_scalars': avg_test_scalars}
            torch.save(checkpoint_data, "{}/best.ckpt".format(args.saveckpt))
        save_scalars(logger, 'fulltest', avg_test_scalars, global_step)
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()

    print('Train completed. MAX epoch %d total test error = %.5f' % (bestepoch, error))


def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_low = disp_gt_low.cuda()
    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)

    disp_gtR = bulidDispR(disp_gt)
    maskR = (disp_gtR < args.maxdisp) & (disp_gtR > 0)
    disp_gtR_low = F.interpolate(disp_gtR, disp_ests[2].shape[1:], mode='nearest')
    maskR_low = (disp_gtR_low < args.maxdisp) & (disp_gtR_low > 0)

    disp_gts = [disp_gt, disp_gtR.squeeze(1), disp_gt_low, disp_gtR_low.squeeze(1)]
    masks = [mask, maskR.squeeze(1), mask_low, maskR_low.squeeze(1)]

    loss = model_loss_train(disp_ests, disp_gts, masks)

    disp_ests_final = [disp_ests[0]]

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests_final]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests_final]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests_final]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests_final]
            scalar_outputs["Thres4"] = [Thres_metric(disp_est, disp_gt, mask, 4.0) for disp_est in disp_ests_final]
            scalar_outputs["Thres5"] = [Thres_metric(disp_est, disp_gt, mask, 5.0) for disp_est in disp_ests_final]

    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)


@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    masks = [mask]
    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gts, masks)

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    scalar_outputs["Thres4"] = [Thres_metric(disp_est, disp_gt, mask, 4.0) for disp_est in disp_ests]
    scalar_outputs["Thres5"] = [Thres_metric(disp_est, disp_gt, mask, 5.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    train()
