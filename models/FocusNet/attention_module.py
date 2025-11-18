import torch
import torch.nn as nn
from .layers import conv
import torch.nn.functional as F

class FocusAttention(nn.Module):
    def __init__(self, in_channel, channel=32):
        super(FocusAttention, self).__init__()
        self.channel = channel

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, 1, 1)

        self.attention_path1 = Attention()
        self.attention_path2 = Attention()

        self.q = nn.Linear(channel, channel, bias=True)
        self.query_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(1, 1, channel), mean=0, std=0.02))
        self.temperature = nn.Parameter(torch.log((torch.ones(1, 1, 1) / 0.24).exp() - 1)) #Initialize softplus(temperature) to 1/0.24.

        self.kv = nn.Linear(channel, channel * 2, bias=False)

        self.window_size = 3
        self.local_len = self.window_size ** 2

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(1, self.local_len), mean=0,
                                  std=0.0004))

        self.sr = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(channel)
        self.act = nn.GELU()

        self.ca = ChannelAttention(channel)
        self.translayer = nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, padding=0)

        self.attn_drop = nn.Dropout(0.)
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(1, channel, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(1, 1, self.local_len))

        self.unfold = nn.Unfold(kernel_size=self.window_size, padding=self.window_size // 2, stride=1)

        self.proj = nn.Linear(channel, channel)
        self.proj_drop = nn.Dropout(0.)

    def forward(self, f, x, seg_map):
        b, c, h, w = f.shape
        n = h * w
        x = self.translayer(x)
        x = F.interpolate(x, size=f.shape[-2:], mode='bilinear', align_corners=False)

        f = f.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)

        # seq_length_scale = torch.log(torch.as_tensor(n, device=f.device))
        # padding_mask = None
        attn_map = F.unfold(torch.ones([1, 1, h, w], device=f.device), self.window_size,
                            dilation=1, padding=(self.window_size // 2, self.window_size // 2), stride=1)
        local_seq_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
        # padding_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
        seq_length_scale = torch.log(local_seq_length + h * w)

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(b, n, 1, self.channel).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        # Compute local similarity
        k_local, v_local = self.kv(f).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(b, n, 1, self.channel), dim=-1).reshape(b, n, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(b, -1, h, w)
        k_local, v_local = self.unfold(kv_local).reshape(
            b, 2, self.channel, self.local_len, n).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        # attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
        #               + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(padding_mask, float('-inf'))
        attn_local = (q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2)

        # Compute pooled similarity
        sr_ratio = 2
        pool_h, pool_w = h // sr_ratio, w // sr_ratio
        pool_len = pool_h * pool_w

        f_ = f.permute(0, 2, 1).reshape(b, -1, h, w).contiguous()
        f_ = F.adaptive_avg_pool2d(self.act(self.sr(f_)), (pool_h, pool_w)).reshape(b, -1, pool_len).permute(0, 2, 1)
        f_ = self.norm(f_)
        kv_pool = self.kv(f_).reshape(b, pool_len, 2, self.channel).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)
        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1)

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(
            -2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        context = (x_local + x_pool).transpose(1, 2).reshape(b, n, c)

        context = self.proj(context)
        context = self.proj_drop(context)

        context = context.reshape(b, c, h, w)

        f = f.reshape(b, c, h, w)
        x = x.reshape(b, c, h, w)
        f_ca = self.ca(f) * f
        res = context * f_ca * x

        res = self.conv_out1(res)
        res = self.conv_out2(res)
        out = self.conv_out3(res)
        res_out = out
        out = out + seg_map
        return x, out, res_out


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def forward(self, x, context):
        b, _, _, _ = x
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        return context

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

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

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
