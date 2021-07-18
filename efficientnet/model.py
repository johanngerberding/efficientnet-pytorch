from albumentations.augmentations.bbox_utils import calculate_bbox_area
import torch
import torch.nn as nn
import numpy as np 

from functools import reduce
from operator import __add__


class Conv2dSamePadding(nn.Conv2d):
    "https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6"
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class SEBlock(nn.Module):
    """Paper: https://arxiv.org/pdf/1709.01507.pdf"""
    def __init__(self, in_channels, reduction_ratio):
        super().__init__()
        self.channels = in_channels
        self.r = reduction_ratio
        assert self.r > 0
        self.squeeze = int(self.channels / self.r)
        self.se = nn.Sequential(nn.Linear(self.channels, self.squeeze),
                                nn.ReLU(),
                                nn.Linear(self.squeeze, self.channels),
                                nn.Sigmoid())

    def forward(self, x):
        x_ = x.mean(dim=(3,2))
        x_ = self.se(x_)
        # channel-wise multiplication
        x_ = x_.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x,x_)
        return x


class MBConv(nn.Module):
    """Paper: https://arxiv.org/pdf/1801.04381.pdf"""
    def __init__(self, in_channels, out_channels, filter_size, expand_ratio, stride, se_ratio, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = filter_size
        self.t = expand_ratio
        self.s = stride
        self.se_ratio = se_ratio
        self.padding = padding
        middle_dim = self.in_channels * self.t

        # MBConv1
        if self.t == 1:
            self.conv = nn.Sequential(nn.Conv2d(middle_dim, middle_dim, self.k, self.s, self.padding, groups=middle_dim, bias=False),
                                      nn.BatchNorm2d(middle_dim),
                                      nn.SiLU(inplace=True),
                                      #Conv2dSamePadding(middle_dim, self.out_channels, 1, bias=False),
                                      nn.Conv2d(middle_dim, self.out_channels, 1, bias=False),
                                      nn.BatchNorm2d(self.out_channels))
        # MBConv6
        else:
            self.conv = nn.Sequential(nn.Conv2d(self.in_channels, middle_dim, 1, bias=False),
                                      nn.BatchNorm2d(middle_dim),
                                      nn.SiLU(inplace=True),
                                      Conv2dSamePadding(middle_dim, middle_dim, self.k, self.s, groups=middle_dim, bias=False),
                                      #nn.Conv2d(middle_dim, middle_dim, self.k, self.s, padding, groups=middle_dim, bias=False),
                                      nn.BatchNorm2d(middle_dim),
                                      nn.SiLU(inplace=True),
                                      nn.Conv2d(middle_dim, self.out_channels, 1, bias=False),
                                      nn.BatchNorm2d(self.out_channels))

        self.se = SEBlock(self.out_channels, self.se_ratio)


    def forward(self, x):
        if self.s == 1 and self.in_channels == self.out_channels:
            x_ = self.se(self.conv(x))
            return x + x_
        else:
            return self.se(self.conv(x))



class EfficientNet(nn.Module):
    def __init__(self, config, num_classes,  name='efficientnet-b0'):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.dropout = self.config['dropout']
        self.in_channels = 3
        self.out_channels = 1280
        self.stages_config = self.config['stages']

        self.stage1 = nn.Sequential(Conv2dSamePadding(self.in_channels, 32, 3, stride=2),
                                    nn.BatchNorm2d(32),
                                    nn.SiLU(inplace=True))
        self.modules = [self._create_stage(params) for params in self.stages_config]
        self.stages = nn.Sequential(*self.modules)
        self.final = nn.Sequential(Conv2dSamePadding(320, self.out_channels, 1),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.AdaptiveAvgPool2d(1),
                                   nn.Flatten(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.out_channels, self.num_classes))


    def forward(self, x):
        x = self.stage1(x)
        x = self.stages(x)
        x = self.final(x)
        return x

    def _create_stage(self, params):
        expand_ratio, filter_size, num_repeats, in_channels, out_channels, stride, padding, se_ratio = params
        modules = []
        for i in range(num_repeats):
            if i == (num_repeats-1):
                modules.append(MBConv(in_channels, out_channels, filter_size, expand_ratio, stride, se_ratio, 1))
            else:
                modules.append(MBConv(in_channels, in_channels, filter_size, expand_ratio, 1, se_ratio, padding))

        return nn.Sequential(*modules)
