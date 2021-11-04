import torch.nn as nn
import numpy as np
from convolution import Conv2dBlock, DepthwiseConv2dBlock, Conv2dStaticSamePadding
from utils import *


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25, bias=True, act_fn='mish', static_padding=False, image_size=None):
        super().__init__()
        squeezed_channels = max(int(in_channels * se_ratio), 1)
        if static_padding:
            self.reduce = Conv2dStaticSamePadding(in_channels, squeezed_channels, 1, image_size=image_size, bias=bias)
        else:
            self.reduce = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1, bias=bias)
        self.act_fn = get_act_fn(act_fn)
        if static_padding:
            self.expand = Conv2dStaticSamePadding(squeezed_channels, in_channels, 1, image_size=image_size, bias=bias)
        else:
            self.expand = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self.reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.expand(x_se)
        x = x * x_se.sigmoid()
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 momentum=0.1, eps=1e-5, ksize=3,
                 stride=1, expand_ratio=1, se_ratio=0.25,
                 drop_ratio=0.2, act_fn='mish', image_size=224):
        super().__init__()
        self.skip_connect = stride == 1 and in_channels == out_channels
        self.drop_ratio = drop_ratio
        self.act_fn = get_act_fn(act_fn)
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        expand_channels = in_channels * expand_ratio
        if not isinstance(image_size, int):
            image_size = np.array(image_size)
        if expand_ratio != 1:
            self.expand_conv = Conv2dBlock(in_channels, expand_channels, 1, bias=False, momentum=momentum, eps=eps, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        self.depthwise_conv = DepthwiseConv2dBlock(expand_channels, ksize, stride=stride, momentum=momentum, eps=eps, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        image_size = np.ceil(image_size / stride).astype(int)
        if se_ratio is not None:
            self.se = SqueezeExcitation(expand_channels, se_ratio=se_ratio, act_fn=self.act_fn, static_padding=True, image_size=image_size)
        self.project_conv = Conv2dBlock(expand_channels, out_channels, 1, bias=False, momentum=momentum, eps=eps, static_padding=True, image_size=image_size)

    def forward(self, x):
        x_in = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se_ratio is not None:
            x = self.se(x)
        x = self.project_conv(x)
        if self.skip_connect:
            x = drop_connect(x, self.drop_ratio, training=self.training) + x_in
        return x


class MBConvForRelativeAttention(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, out_channels,
                 momentum=0.1, eps=1e-5, ksize=3, expand_ratio=1,
                 se_ratio=0.25, drop_ratio=0.1, act_fn='mish',
                 use_downsampling=False, **kwargs):
        super().__init__()
        self.use_downsampling = use_downsampling
        self.drop_ratio = drop_ratio
        self.norm = nn.BatchNorm2d(in_channels)
        self.mbconv = MBConv(in_channels, out_channels, momentum=momentum, eps=eps,
                             ksize=ksize, stride=2 if use_downsampling else 1,
                             expand_ratio=expand_ratio, se_ratio=se_ratio,
                             drop_ratio=drop_ratio, act_fn=act_fn, image_size=(inp_h, inp_w))
        if use_downsampling:
            self.pool = nn.MaxPool2d((2, 2))
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.use_downsampling:
            x_downsample = self.pool(x)
            x_downsample = self.conv(x_downsample)
        else:
            x_downsample = x
        x = self.norm(x)
        x = self.mbconv(x)
        x = drop_connect(x, self.drop_ratio, training=self.training)
        x = x_downsample + x
        return x
