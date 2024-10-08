from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d: # 3d
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else: # 2d
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x) #, inplace=True) # LeakyReLU
        return x


class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

def groupwise_difference(fea1, fea2, num_groups):
    B, G, C, H, W = fea1.shape
    cost = torch.pow((fea1 - fea2), 2).sum(2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_struct_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, G, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_difference(refimg_fea[:, :, :, :, i:], targetimg_fea[:, :, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_difference(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def build_bi_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    l_volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    r_volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            concat = torch.cat((refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i]), dim=1)
            l_volume[:, :, i, :, i:] = concat
            r_volume[:, :, i, :, :-i] = concat
        else:
            concat = torch.cat((refimg_fea, targetimg_fea), dim=1)
            l_volume[:, :, i, :, :] = concat
            r_volume[:, :, i, :, :] = concat
    l_volume = l_volume.contiguous()
    r_volume = r_volume.contiguous()
    return l_volume, r_volume


def build_bi_add_feat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    l_volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    r_volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            add = refimg_fea[:, :, :, i:] + targetimg_fea[:, :, :, :-i]
            l_volume[:, :, i, :, i:] = add
            r_volume[:, :, i, :, :-i] = add
        else:
            add = refimg_fea + targetimg_fea
            l_volume[:, :, i, :, :] = add
            r_volume[:, :, i, :, :] = add
    l_volume = l_volume.contiguous()
    r_volume = r_volume.contiguous()
    return l_volume, r_volume


def build_bi_tar_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    l_volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    r_volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            l_volume[:, :, i, :, i:] = targetimg_fea[:, :, :, :-i]
            r_volume[:, :, i, :, :-i] = refimg_fea[:, :, :, i:]
        else:
            l_volume[:, :, i, :, :] = targetimg_fea
            r_volume[:, :, i, :, :] = refimg_fea
    l_volume = l_volume.contiguous()
    r_volume = r_volume.contiguous()
    return l_volume, r_volume


def build_bi_tar_volume_step(refimg_fea, targetimg_fea, maxdisp, disp_step):
    B, C, H, W = refimg_fea.shape
    l_volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    r_volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    for i in range(maxdisp):
        j = i * disp_step
        if i > 0:
            l_volume[:, :, i, :, j:] = targetimg_fea[:, :, :, :-j]
            r_volume[:, :, i, :, :-j] = refimg_fea[:, :, :, j:]
        else:
            l_volume[:, :, i, :, :] = targetimg_fea
            r_volume[:, :, i, :, :] = refimg_fea
    l_volume = l_volume.contiguous()
    r_volume = r_volume.contiguous()
    return l_volume, r_volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])

    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume_norm(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation_norm(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def norm_correlation(fea1, fea2):

    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost


def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume


def build_bi_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    l_volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    r_volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            norm_corr = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
            l_volume[:, :, i, :, i:] = norm_corr
            r_volume[:, :, i, :, :-i] = norm_corr
        else:
            norm_corr = norm_correlation(refimg_fea, targetimg_fea)
            l_volume[:, :, i, :, :] = norm_corr
            r_volume[:, :, i, :, :] = norm_corr
    l_volume = l_volume.contiguous()
    r_volume = r_volume.contiguous()
    return l_volume, r_volume


def build_bi_norm_correlation_volume_step(refimg_fea, targetimg_fea, maxdisp, disp_step):
    B, C, H, W = refimg_fea.shape
    l_volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    r_volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        j = i * disp_step
        if i > 0:
            norm_corr = norm_correlation(refimg_fea[:, :, :, j:], targetimg_fea[:, :, :, :-j])
            l_volume[:, :, i, :, j:] = norm_corr
            r_volume[:, :, i, :, :-j] = norm_corr
        else:
            norm_corr = norm_correlation(refimg_fea, targetimg_fea)
            l_volume[:, :, i, :, :] = norm_corr
            r_volume[:, :, i, :, :] = norm_corr
    l_volume = l_volume.contiguous()
    r_volume = r_volume.contiguous()
    return l_volume, r_volume

def SpatialTransformer_grid(x, y, disp_range_samples):
    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([
        torch.arange(0, height, dtype=x.dtype, device=x.device),
        torch.arange(0, width, dtype=x.dtype, device=x.device)]) # (H *W)
    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1) # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples
    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0 # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) # (B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear', padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width) # (B, C, D, H, W)

    return y_warped


def context_upsample(depth_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = depth_low.shape

    depth_unfold = F.unfold(depth_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    depth_unfold = F.interpolate(depth_unfold, (h*4, w*4), mode='nearest').reshape(b, 9, h*4, w*4)

    depth = (depth_unfold * up_weights).sum(1)

    return depth


def regression_topk(cost, disparity_samples, k):
    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred


def regression_soft_argmin(cost, disparity_samples):
    assert len(cost.shape) == 4, [cost.shape]
    prob = F.softmax(cost, 1)
    return torch.sum(disparity_samples * prob, dim=1, keepdim=True)


def bulidCostL(costR):
    assert costR.dim() == 5, [costR.shape]
    B, C, D, H, W = costR.size()
    costL = costR.new_zeros([B, C, D, H, W])
    for d in range(D):
        if d > 0:
            costL[:, :, d, :, d:] = costR[:, :, d, :, :-d]
        else:
            costL[:, :, d, :, :] = costR[:, :, d, :, :]
    costL = costL.contiguous()
    return costL


def bulidCostR(costL):
    assert costL.dim() == 5, [costL.shape]
    B, C, D, H, W = costL.size()
    costR = costL.new_zeros([B, C, D, H, W])
    for d in range(D):
        if d > 0:
            costR[:, :, d, :, :-d] = costL[:, :, d, :, d:]
        else:
            costR[:, :, d, :, :] = costL[:, :, d, :, :]
    costR = costR.contiguous()
    return costR


def bulidDispL(dispR):
    if len(dispR.shape) != 4: dispR = dispR.unsqueeze(1)
    assert dispR.shape[1] == 1, [dispR.shape]
    B, C, H, W = dispR.size()

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    xx = xx.view(1, 1, H, W).repeat(B, C, 1, 1) # (B,C,H,W)
    if dispR.is_cuda: xx = xx.cuda()
    dispL = torch.zeros_like(dispR, device=dispR.device)

    index = xx + dispR
    index = torch.clamp(index, 0, W-1).to(dtype=torch.int64)

    dispL = torch.scatter(dispL, -1, index, dispR)
    return dispL


def bulidDispR(dispL):
    if len(dispL.shape) != 4: dispL = dispL.unsqueeze(1)
    assert dispL.shape[1] == 1, [dispL.shape]
    B, C, H, W = dispL.size()

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    xx = xx.view(1, 1, H, W).repeat(B, C, 1, 1) # (B,C,H,W)
    if dispL.is_cuda: xx = xx.cuda()
    dispR = torch.zeros_like(dispL, device=dispL.device)

    index = xx - dispL
    index = torch.clamp(index, 0, W-1).to(dtype=torch.int64)

    dispR = torch.scatter(dispR, -1, index, dispL)
    return dispR


def bulidImgL(imgR, dispR):
    B, C, H, W = imgR.size()
    if len(dispR.shape) != 4: dispR = dispR.unsqueeze(1).repeat(1, C, 1 , 1)
    elif dispR.shape[1] == 1: dispR = dispR.repeat(1, C, 1 , 1)
    assert imgR.shape == dispR.shape, [imgR.shape, dispR.shape]

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    xx = xx.view(1, 1, H, W).repeat(B, C, 1, 1) # (B,C,H,W)
    if imgR.is_cuda: xx = xx.cuda()
    imgL = torch.zeros_like(imgR, device=imgR.device, requires_grad=True)

    index = xx + dispR
    index = torch.clamp(index, 0, W-1).to(dtype=torch.int64)

    imgL = torch.scatter(imgL, -1, index, imgR)
    return imgL


def bulidImgR(imgL, disp):
    B, C, H, W = imgL.size()
    if len(disp.shape) != 4: disp = disp.unsqueeze(1).repeat(1, C, 1 , 1)
    elif disp.shape[1] == 1: disp = disp.repeat(1, C, 1 , 1)
    assert imgL.shape == disp.shape, [imgL.shape, disp.shape]

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    xx = xx.view(1, 1, H, W).repeat(B, C, 1, 1) # (B,C,H,W)
    if imgL.is_cuda: xx = xx.cuda()
    imR = torch.zeros_like(imgL, device=imgL.device, requires_grad=True)

    index = xx - disp
    index = torch.clamp(index, 0, W-1).to(dtype=torch.int64)

    imR = torch.scatter(imR, -1, index, imgL)
    return imR


def warp_disp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border', align_corners=True)

    return output


class RefineBasicBlock(nn.Module):
    def __init__(self, planes, kernel_size=3, stride=1, dilation=1, leaky_relu=True):
        """StereoNet uses leaky relu (alpha = 0.2)"""
        super(RefineBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding='same', dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding='same', dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)

        return out


class MultiConv(nn.Module):
    def __init__(self, in_channels=6, kernel_list=[3,5,7]):
        super(MultiConv, self).__init__()

        self.kernel_list = kernel_list
        self.kernel_blocks = nn.ModuleList()

        for kernel in self.kernel_list:
            self.kernel_blocks.append(RefineBasicBlock(in_channels, kernel_size=kernel, stride=1))

        self.agg = nn.Sequential(
            BasicConv((len(kernel_list)+1)*in_channels, in_channels, kernel_size=5, padding=2, stride=1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
            )

    def forward(self, x):
        out = x
        for multi_kernel in self.kernel_blocks:
            out = torch.cat((out, multi_kernel(x)), dim=1)

        return self.agg(out) + x # [B, in_channels, H, W]


class StereoDRNetRefinement(nn.Module):
    def __init__(self):
        super(StereoDRNetRefinement, self).__init__()

        in_channels = 6

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True)
            )

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(RefineBasicBlock(32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, disp, left_img, right_img):
        assert disp.dim() == 3, [disp.shape]
        assert left_img.size(-1) == disp.size(-1), [left_img.size(-1), disp.size(-1)]

        warpL = warp_disp(right_img, -disp)
        error = warpL - left_img

        concat1 = torch.cat((error, left_img), dim=1)

        conv1 = self.conv1(concat1) # [B, 16, H, W]
        conv2 = self.conv2(disp.unsqueeze(1)) # [B, 16, H, W]
        concat2 = torch.cat((conv1, conv2), dim=1)

        out = self.dilated_blocks(concat2)
        residual_disp = self.final_conv(out)

        disp = F.relu(disp.unsqueeze(1) + residual_disp, inplace=True) # [B, H, W]

        return disp
