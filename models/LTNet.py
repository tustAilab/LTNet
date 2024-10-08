from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .submodule import *
import math
import timm


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()


    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]


        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1


        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]


class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR

        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]


class Context_Geometry_Fusion(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(Context_Geometry_Fusion, self).__init__()

        self.semantic = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1)
            )

        self.att = nn.Sequential(
            BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1, dilation=1),
            nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False)
            )

        self.agg = BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1, dilation=1)

        self.weight_init()

    def forward(self, cv, feat):
        feat = self.semantic(feat).unsqueeze(2)
        att = self.att(feat+cv)
        cv = torch.sigmoid(att)*feat + cv
        cv = self.agg(cv)

        return cv


class hourglass_fusion(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1)
            )

        self.conv2 = nn.Sequential(
            BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1)
            )

        self.conv3 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1)
            )

        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1)
            )

        self.agg_1 = nn.Sequential(
            BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1)
            )

        self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 160)
        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192)
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = self.CGF_32(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2 = self.CGF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv1 = self.CGF_8(conv1, imgs[1])
        conv = self.conv1_up(conv1)

        return conv, conv2_up, conv3_up


class Consistency_Refinement(nn.Module):
    def __init__(self):
        super(Consistency_Refinement, self).__init__()
        hide_channel = 8

        self.stem = BasicConv(1, 6, kernel_size=1, padding=0, stride=1)

        self.agg_corr = nn.Sequential(
            BasicConv(2, hide_channel, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(hide_channel, hide_channel, kernel_size=1, stride=1, padding=0, bias=False)
            )

        self.mult_conv = nn.ModuleList()
        for i in range(3):
            self.mult_conv.append(MultiConv(hide_channel, kernel_list=[3, 5, 7]))
        self.mult_conv = nn.Sequential(*self.mult_conv)

        self.res_conv = nn.Conv2d(hide_channel, 1, 3, 1, 1)

    def forward(self, ref_disp, target_disp):
        warp_ref = bulidDispL(target_disp)
        mask = warp_ref / (warp_ref + 1e-05)
        consis = norm_correlation(self.stem(ref_disp), self.stem(warp_ref))

        attn = self.agg_corr(torch.concat((ref_disp, consis), dim=1)) * mask
        residual = self.mult_conv(ref_disp * F.tanh(attn))
        residual = self.res_conv(residual)

        ref_disp = F.relu(ref_disp + residual, inplace=True)

        return ref_disp

class LTNet(nn.Module):
    def __init__(self, maxdisp):
        super(LTNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature = Feature()
        self.feature_up = FeatUp()
        cost_channels = 8

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.corr_stem = BasicConv(1, cost_channels, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.semantic = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, cost_channels, kernel_size=1, padding=0, stride=1, bias=False)
            )
        self.agg = BasicConv(cost_channels, cost_channels, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)

        self.hourglass_fusion = hourglass_fusion(cost_channels)

        self.consis_refinement = Consistency_Refinement()

        self.spx = nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.refinement = StereoDRNetRefinement()


        if self.training:
            self.mid_8x = BasicConv(2*cost_channels, 1, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
            self.mid_16x = nn.Sequential(
                BasicConv(4*cost_channels, 2*cost_channels, deconv=True, is_3d=True, bn=False, relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2)),
                nn.ConvTranspose3d(2*cost_channels, 1, 4, 2, 1, bias=False)
            )

    def forward(self, left, right):

        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)

        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))


        corr_volume_l, corr_volume_r = build_bi_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        corr_volume_l = self.corr_stem(corr_volume_l)
        corr_volume_r = self.corr_stem(corr_volume_r)

        feat_l = self.semantic(features_left[0])
        feat_r = self.semantic(features_right[0])
        tar_volume_l, tar_volume_r = build_bi_tar_volume(feat_l, feat_r, self.maxdisp//4)

        volume_l = self.agg(tar_volume_l * corr_volume_l)
        volume_r = self.agg(tar_volume_r * corr_volume_r)


        cost_l, cost_8x_l, cost_16x_l = self.hourglass_fusion(volume_l, features_left)
        cost_r, cost_8x_r, cost_16x_r = self.hourglass_fusion(volume_r, features_right)


        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost_l.dtype, device=cost_l.device)
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost_l.shape[0], 1, cost_l.shape[3], cost_l.shape[4])
        pred_l = regression_topk(cost_l.squeeze(1), disp_samples, 2)
        pred_r = regression_topk(cost_r.squeeze(1), disp_samples, 2)


        pred = self.consis_refinement(pred_l, pred_r)


        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        pred_up = context_upsample(pred, spx_pred) * 4


        pred_up = self.refinement(pred_up, left, right)

        if self.training:

            cost_8x_l = self.mid_8x(cost_8x_l)
            cost_8x_r = self.mid_8x(cost_8x_r)
            pred_8x_l = regression_topk(cost_8x_l.squeeze(1), disp_samples, 2)
            pred_8x_r = regression_topk(cost_8x_r.squeeze(1), disp_samples, 2)


            cost_16x_l = self.mid_16x(cost_16x_l)
            cost_16x_r = self.mid_16x(cost_16x_r)
            pred_16x_l = regression_topk(cost_16x_l.squeeze(1), disp_samples, 2)
            pred_16x_r = regression_topk(cost_16x_r.squeeze(1), disp_samples, 2)


            return [pred_up.squeeze(1), pred.squeeze(1)*4, pred_8x_l.squeeze(1)*4, pred_16x_l.squeeze(1)*4, pred_8x_r.squeeze(1)*4, pred_16x_r.squeeze(1)*4]
        else:

            return [pred_up.squeeze(1)]
