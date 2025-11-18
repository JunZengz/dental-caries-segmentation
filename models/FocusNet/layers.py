import torch.nn as nn
import torch
from torchvision.ops import deform_conv2d
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU (inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DWConv_Mulit(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_Mulit, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv_Mulit(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        #self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


#通道注意力
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output



class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            # mask=modulator,
            stride=self.stride,
        )
        return x


##################### Deformable
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


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformable, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp2 = _ASPPModuleDeformable(in_channels, self.in_channelster, 3, padding=0)
        self.aspp3 = _ASPPModuleDeformable(in_channels, self.in_channelster, 5, padding=0)
        self.aspp4 = _ASPPModuleDeformable(in_channels, self.in_channelster, 7, padding=0)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.in_channelster),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.in_channelster * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x2 = self.ret(x2, x1)
        x3 = self.ret(x3, x1)
        x4 = self.ret(x4, x1)
        x5 = self.ret(x5, x1)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class ASPPDeformableV2(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV2, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp2 = _ASPPModuleDeformable(in_channels, self.in_channelster, 3, padding=0)
        self.aspp3 = _ASPPModuleDeformable(in_channels, self.in_channelster, 5, padding=0)
        self.aspp4 = _ASPPModuleDeformable(in_channels, self.in_channelster, 7, padding=0)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(self.in_channelster, self.in_channelster, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.in_channelster),
                                             nn.ReLU(inplace=True))
        self.global_max_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                             nn.Conv2d(self.in_channelster, self.in_channelster, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.in_channelster),
                                             nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(self.in_channelster * 4, self.in_channelster, 1, bias=False)
        self.conv2 = nn.Conv2d(self.in_channelster, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x2 = self.ret(x2, x1)
        x3 = self.ret(x3, x1)
        x4 = self.ret(x4, x1)
        x_ = torch.cat((x1, x2, x3, x4), dim=1)
        x_ = self.conv1(x_)

        x5 = self.global_avg_pool(x_)
        x6 = self.global_max_pool(x_)
        s = self.sigmoid(x5+x6)
        x = x_ * s.expand_as(x_)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class ASPPDeformableV3(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV3, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp2 = _ASPPModuleDeformable(in_channels, self.in_channelster, 3, padding=0)
        self.aspp3 = _ASPPModuleDeformable(in_channels, self.in_channelster, 5, padding=0)
        self.aspp4 = _ASPPModuleDeformable(in_channels, self.in_channelster, 7, padding=0)

        self.eca = eca_layer(self.in_channelster)

        self.conv1 = nn.Conv2d(self.in_channelster * 4, self.in_channelster, 1, bias=False)
        self.conv2 = nn.Conv2d(self.in_channelster, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
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


class ASPPDeformableV4(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV4, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels, self.in_channelster, (1, 5), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (5, 1), padding=0))
        self.aspp4 = nn.Sequential(_ASPPModuleDeformable(in_channels, self.in_channelster, (1, 7), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (7, 1), padding=0))

        self.eca = eca_layer(self.in_channelster)

        self.conv1 = nn.Conv2d(self.in_channelster * 4, self.in_channelster, 1, bias=False)
        self.conv2 = nn.Conv2d(self.in_channelster, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
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




class ASPPDeformableV5(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV5, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 5), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (5, 1), padding=0))
        self.aspp4 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 7), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (7, 1), padding=0))

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


class ASPPDeformableV6(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV6, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 7), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (7, 1), padding=0))
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels // 4, self.in_channelster, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.in_channelster),
                                             nn.ReLU(inplace=True))
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
        x4 = self.global_avg_pool(x_4)
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



class ASPPDeformableV7(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV7, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 3), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 5), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (5, 1), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 3), padding=0))
        self.aspp4 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 7), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (7, 1), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 3), padding=0))

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



class ASPPDeformableV8(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV8, self).__init__()
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



class ASPPDeformableV9(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV9, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp4 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 1), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))

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



class ASPPDeformableV10(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV10, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (5, 5), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (5, 5), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (5, 5), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp4 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4, self.in_channelster, (5, 5), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))

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




class ASPPDeformableV11(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPDeformableV11, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4,  self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4,  self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp3 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4,  self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))
        self.aspp4 = nn.Sequential(_ASPPModuleDeformable(in_channels // 4,  self.in_channelster, (3, 3), padding=0),
                                   _ASPPModuleDeformable(self.in_channelster, self.in_channelster, (1, 1), padding=0))

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



class ASPPBasicV5(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ASPPBasicV5, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale
        self.in_channels = in_channels
        self.aspp1 = nn.Sequential(BasicConv2d(in_channels // 4, self.in_channelster, (1, 1), padding=0))
        self.aspp2 = nn.Sequential(BasicConv2d(in_channels // 4, self.in_channelster, (1, 3), padding=0),
                                   BasicConv2d(self.in_channelster, self.in_channelster, (3, 1), padding=0))
        self.aspp3 = nn.Sequential(BasicConv2d(in_channels // 4, self.in_channelster, (1, 5), padding=0),
                                   BasicConv2d(self.in_channelster, self.in_channelster, (5, 1), padding=0))
        self.aspp4 = nn.Sequential(BasicConv2d(in_channels // 4, self.in_channelster, (1, 7), padding=0),
                                   BasicConv2d(self.in_channelster, self.in_channelster, (7, 1), padding=0))

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
