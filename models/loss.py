import torch.nn.functional as F
from .submodule import *


def model_loss_test(disp_ests, disp_gts, img_masks):
    weights = [1.0]
    all_losses = []
    for disp_est, disp_gt, weight, mask_img in zip(disp_ests, disp_gts, weights, img_masks):
        all_losses.append(weight * F.l1_loss(disp_est[mask_img], disp_gt[mask_img], reduction='mean'))
    return sum(all_losses)



def model_loss_train(disp_ests, disp_gts, img_masks):
    all_losses = []


    weights = [1.0, 0.3]
    disps = disp_ests[:2]
    gts = [disp_gts[0], disp_gts[2]]
    masks = [img_masks[0], img_masks[2]]

    for disp, gt, weight, mask in zip(disps, gts, weights, masks):
        all_losses.append(weight * F.smooth_l1_loss(disp[mask], gt[mask], reduction='mean'))


    weights = [0.2, 0.1, 0.2, 0.1]
    disps = disp_ests[2:6]
    gts = [disp_gts[2], disp_gts[2], disp_gts[3], disp_gts[3]]
    masks = [img_masks[2], img_masks[2], img_masks[3], img_masks[3]]

    for disp, gt, weight, mask in zip(disps, gts, weights, masks):
        all_losses.append(weight * F.smooth_l1_loss(disp[mask], gt[mask], reduction='mean'))

    return sum(all_losses)
