import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import time as tm

import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor

from .utils import (num_param, Concat2d, MultiBatchNorm2d,
                    RepVGGBlock_train, RepVGGBlock_train_shared_bn, model_info,
                    trace_net, round)
from .repvgg import RepVGGBlock

from timm.models.layers.classifier import ClassifierHead


class SimpNet(nn.Module):
    def __init__(self, planes,
                 num_blocks,
                 dropout_lin,
                 additional_branches=[]):
        """
        :param planes: number of planes in each stream
        :param num_blocks: number of blocks in each stream
        :param dropout_lin: adds dropout to the final linear block
        """
        super().__init__()
        print(f"Initializing ParNet with planes: {planes}")

        self.num_classes = 1000
        block = RepVGGOur

        last_planes = planes[-1]
        planes = planes[0:-1]
        strides = [2] * len(planes)
        assert (num_blocks[-1] == 1)
        assert (num_blocks[-2] != 1)
        num_blocks = num_blocks[0:-1]

        # inits
        self.inits = nn.ModuleList()
        in_planes = min(64, planes[0])
        inits = nn.Sequential(
            block(
                3, in_planes, stride=2,
                additional_branches=additional_branches),
            block(
                in_planes, planes[0], stride=2,
                additional_branches=additional_branches))

        self.inits.append(inits)
        for i, (stride, in_plane, out_plane) in enumerate(
                zip(strides[1:], planes[0:-1], planes[1:])):
            self.inits.append(
                block(
                    in_plane * block.expansion,
                    out_plane * block.expansion,
                    stride,
                    additional_branches=additional_branches))

        # streams
        self.streams = nn.ModuleList()

        def stream_block(stream_id, i, plane):
            _args = {
                'stride': 1,
                'se_block': True,
                'additional_branches': additional_branches,
            }
            out_block = block(plane, plane, kernel_size=3, **_args)

            return out_block

        for stream_id, (num_block, plane) in enumerate(
                zip(num_blocks, planes)):
            stream = nn.ModuleList()

            for i in range(num_block - 1):
                stream.append(
                    stream_block(stream_id, i, plane * block.expansion))
            self.streams.append(nn.Sequential(*stream))

        # downsamples_2
        self.downsamples_2 = nn.ModuleList()
        in_planes = planes[0:-1]
        out_planes = planes[1:]
        for i, (stride, in_plane, out_plane) in enumerate(zip(
                strides[1:], in_planes, out_planes)):
            if i == 0:
                self.downsamples_2.append(
                    block(in_plane * block.expansion,
                          out_plane * block.expansion,
                          stride,
                          kernel_size=3,
                          additional_branches=additional_branches))
            else:
                layer = nn.Sequential(
                    MultiBatchNorm2d(
                        in_plane * block.expansion,
                        in_plane * block.expansion),
                    Concat2d(shuffle=True),
                    block(
                        2 * in_plane * block.expansion,
                        out_plane * block.expansion,
                        stride=2,
                        groups=2, kernel_size=3,
                        additional_branches=additional_branches))
                self.downsamples_2.append(layer)

        # combine
        in_planes_combine = planes[-1]
        combine = [
            MultiBatchNorm2d(
                in_planes_combine * block.expansion,
                in_planes_combine * block.expansion),
            Concat2d(shuffle=True),
            block(
                2 * in_planes_combine * block.expansion,
                in_planes_combine * block.expansion,
                stride=1,
                groups=2,
                additional_branches=additional_branches),
            block(
                planes[-1], last_planes,
                stride=2,
                additional_branches=additional_branches)
        ]
        self.combine = nn.Sequential(*combine)

        # head
        self.head = ClassifierHead(
            last_planes * block.expansion,
            self.num_classes,
            pool_type='avg',
            drop_rate=dropout_lin)
        self.num_features = last_planes * block.expansion

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        pass

    def forward_features(self, x):
        pass

    def forward(self, img):
        x = img
        x_list = []
        for i, init in enumerate(self.inits):
            x = init(x)
            x_list.append(x)

        y_old = None
        for i, (x, stream) in enumerate(zip(x_list, self.streams)):
            y = stream(x)

            if y_old is None:
                y_old = self.downsamples_2[i](y)
            elif i < len(self.downsamples_2):
                y_old = self.downsamples_2[i]((y, y_old))
            else:
                y_old = (y, y_old)

        out = self.combine(y_old)
        out = self.head(out)

        return out


class RepVGGOur(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 kernel_size=3, se_block=True, additional_branches=[]):
        """
        :param in_planes: True number of planes coming in
        :param planes: True number of planes going out
        """
        super().__init__()

        activation = nn.ReLU()
        if 'swish' in additional_branches:
            activation = nn.SiLU()

        self.block = RepVGGBlock(
            inplanes, planes, kernel_size, stride,
            padding=1, groups=groups, avg_pool=True,
            se_block=se_block,
            activation=activation,
        )

    def forward(self, x):
        out = self.block(x)
        return out


if __name__ == '__main__':
    net = SimpNet(
        'imagenet',
        planes=[round(1 * x) for x in (64, 128, 256, 512)],
        num_blocks=[5, 6, 6, 1],
        dropout_lin=0.0,
        additional_branches=['swish'],
    )
    y = torch.randn(1, 3, 224, 224)
    print(f"Num Parameters: {num_param(net)}")
    # print(net)
    out = net(y)
    trace_net(net, y)
    breakpoint()

