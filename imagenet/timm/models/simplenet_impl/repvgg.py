# Modified from the source: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
import torch.nn as nn
import numpy as np
import torch

from .utils import num_param, trace_net, SE1, channel_shuffle

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', avg_pool=False,
                 se_block=False, activation=nn.ReLU(),
                 ):

        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.padding_mode = padding_mode
        self.se_block = se_block

        assert padding == 1
        padding_11 = padding - 3 // 2

        self.fused = False

        self.dense_groups = groups
        self.nonlinearity = activation

        self.rbr_identity = nn.BatchNorm2d(
            num_features=in_channels) if (
                out_channels == in_channels and
                stride == 1) else None
        self.rbr_dense = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=self.dense_groups) if (kernel_size != 1) else None
        self.rbr_1x1 = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding_11,
            groups=groups)
        if stride == 2 and avg_pool:
            self.rbr_1x1 = nn.Sequential(
                nn.AvgPool2d(2, 2),
                conv_bn(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=1, stride=1, padding=0, groups=groups)
            )

        # updated to reuse code
        self.channel_shuffle = (groups > 1)

        if self.se_block:
            self.se = SE1(
                in_channels, out_channels, g=groups,
                ver=2 if (out_channels != in_channels or stride != 1) else 1)

    def _forward(self, inputs):
        if not self.fused:
            rbr_1x1_output = self.rbr_1x1(inputs)
        else:
            rbr_1x1_output = None

        if self.rbr_dense is None:
            dense_output = 0
        else:
            dense_output = self.rbr_dense(inputs)

        return rbr_1x1_output, dense_output

    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        rbr_1x1_output, drop_path_output = self._forward(inputs)

        if self.se_block:
            if self.rbr_identity is not None:
                id_out = id_out * self.se(id_out)

        if not self.fused:
            out = drop_path_output + rbr_1x1_output + id_out
        else:
            out = drop_path_output + id_out

        if self.se_block and (self.rbr_identity is None):
            out = out * self.se(inputs)

        out = self.nonlinearity(out)

        if self.channel_shuffle:
            out = channel_shuffle(out, self.groups)

        return out

    def fuse_conv_bn(self, conv, bn):
        """
        # n,c,h,w - conv
        # n - bn (scale, bias, mean, var)

        if type(bn) is nn.Identity or type(bn) is None:
            return

        conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        """
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)

        return conv

    def fuse_repvgg_block(self):
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense.conv, self.rbr_dense.bn)

        if isinstance(self.rbr_1x1, nn.Sequential) and isinstance(self.rbr_1x1[0], nn.AvgPool2d):
            self.rbr_1x1[1] = self.fuse_conv_bn(self.rbr_1x1[1].conv, self.rbr_1x1[1].bn)
            rbr_1x1_bias = self.rbr_1x1[1].bias

            weight_1x1_expanded = torch.nn.functional.interpolate(self.rbr_1x1[1].weight, scale_factor=2.0, mode='nearest')
            weight_1x1_expanded = weight_1x1_expanded / 4
            weight_1x1_expanded = torch.nn.functional.pad(weight_1x1_expanded, [1, 0, 1, 0])
        else:
            self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1.conv, self.rbr_1x1.bn)
            rbr_1x1_bias = self.rbr_1x1.bias

            weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias)

        self.rbr_1x1 = nn.Identity()

        self.fused = True
