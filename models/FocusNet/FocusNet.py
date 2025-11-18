import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_module import FocusAttention, ChannelAttention, SpatialAttention
from .layers import BasicConv2d, DeformableConv2d, eca_layer
from .pvtv2 import pvt_v2_b4
from utils.losses import *

def DiceBCELoss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

    
class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class DEM(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(DEM, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0))
        self.aspp4 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0))

        self.eca = eca_layer(self.in_channelster)

        self.conv1 = nn.Conv2d(self.in_channelster * 4, self.in_channelster, 1, bias=False)
        self.conv2 = nn.Conv2d(self.in_channelster, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        x_1, x_2, x_3, x_4 = torch.split(x, self.in_channels // 4, dim=1)
        x1 = self.aspp1(x_1)
        x2 = self.aspp2(x_2)
        x3 = self.aspp3(x_3)
        x4 = self.aspp4(x_4)
        x2 = self.ret(x2, x1)
        x3 = self.ret(x3, x1)
        x4 = self.ret(x4, x1)
        x_ = torch.cat((x1, x2, x3, x4), dim=1)
        x_ = self.conv1(x_)

        x = self.eca(x_)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class CIDM_M(nn.Module):
    def __init__(self, channel):
        super(CIDM_M, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up03 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up04 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv3 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.out_conv = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        xh = x1
        xm = self.conv_upsample1(self.upsample(x1)) * x2    
        xl = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        xm = self.up03(xm)
        xh = self.up04(xh)
        xm_ca = self.ca(xh) * xm  # channel branch
        xm_ca = self.conv1(xm_ca)
        xm_sa = self.sa(xl) * xm  # spatial branch
        xm_sa = self.conv2(xm_sa)
        x = self.conv3(torch.cat((xl, xm_ca, xm_sa, xh), 1))
        out = self.out_conv(x)

        return x, out


class CIDM_A(nn.Module):
    def __init__(self, channel):
        super(CIDM_A, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up03 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up04 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv3 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.out_conv = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        xh = x1
        xm = self.conv_upsample1(self.upsample(x1)) + x2
        xl = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               + self.conv_upsample3(self.upsample(x2)) + x3

        xm = self.up03(xm)
        xh = self.up04(xh)
        xm_ca = self.ca(xh) * xm  # channel branch
        xm_ca = self.conv1(xm_ca)
        xm_sa = self.sa(xl) * xm  # spatial branch
        xm_sa = self.conv2(xm_sa)
        x = self.conv3(torch.cat((xl, xm_ca, xm_sa, xh), 1))
        out = self.out_conv(x)

        return x, out


class FocusNet(nn.Module):
    def __init__(self, channel=32):
        super(FocusNet, self).__init__()

        self.pvt = pvt_v2_b4()  # [64, 128, 320, 512]
        path = 'pretrained_pth/PVT/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.pvt.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.pvt.load_state_dict(model_dict)

        self.Translayer_pvt2 = BasicConv2d(128, channel, 1)
        self.Translayer_pvt3 = BasicConv2d(320, channel, 1)
        self.Translayer_pvt4 = BasicConv2d(512, channel, 1)

        self.translayer_context = BasicConv2d(128, channel, 1)

        self.context = DEM(64, channel)

        self.translayer_context = BasicConv2d(128, channel, 1)

        self.decoder1 = CIDM_M(channel)
        self.decoder2 = CIDM_A(channel)

        self.attention = FocusAttention(channel, channel)

        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        self.loss_fn = DiceFocalLoss(alpha=0.75, gamma=2.0, dice_weight=0.5, focal_weight=0.5)

    def forward(self, sample):
        x = sample['images']
        y = sample['masks']
        base_size = x.shape[-2:]

        # PVTv2
        pvt = self.pvt(x)
        x1_pvt = pvt[0]
        x2_pvt = pvt[1]
        x3_pvt = pvt[2]
        x4_pvt = pvt[3]

        x2_pvt = self.Translayer_pvt2(x2_pvt)
        x3_pvt = self.Translayer_pvt3(x3_pvt)
        x4_pvt = self.Translayer_pvt4(x4_pvt)
        f1, a1 = self.decoder1(x4_pvt, x3_pvt, x2_pvt)
        out1 = self.res(a1, base_size)
        f2, a2 = self.decoder2(x4_pvt, x3_pvt, x2_pvt)
        out2 = self.res(a2, base_size)

        x_t = self.context(x1_pvt)

        f3, a3, res_a3 = self.attention(f1, x_t, a1)
        out3 = self.res(a3, base_size)

        f4, a4, res_a4 = self.attention(f2, x_t, a2)
        out4 = self.res(a4, base_size)

        out = out1 + out2 + out3 + out4

        loss1 = self.loss_fn(out1, y)
        loss2 = self.loss_fn(out2, y)
        loss3 = self.loss_fn(out3, y)
        loss4 = self.loss_fn(out4, y)

        loss = loss1 + loss2 + loss3 + loss4

        return {'prediction': out, 'loss': loss}
