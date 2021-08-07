import torch
import torch.nn as nn
import numpy as np 

from functools import reduce
from operator import __add__

from utils import ModelParams, get_n_params

class Conv2dSamePadding(nn.Conv2d):
    "https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6"
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

"""
First try SE Block based on Paper:

class SEBlock(nn.Module):
    #Paper: https://arxiv.org/pdf/1709.01507.pdf
    def __init__(self, block_in_channels, out_channels, reduction_ratio):
        super().__init__()
        self.block_in_channels = block_in_channels
        self.se_channels = out_channels
        self.r = reduction_ratio
        assert self.r > 0
        self.squeeze = int(self.r * self.block_in_channels)
        self.se = nn.Sequential(nn.Linear(self.se_channels, self.squeeze),
                                nn.SiLU(),
                                nn.Linear(self.squeeze, self.se_channels),
                                nn.Sigmoid())

    def forward(self, x):
        x_ = x.mean(dim=(3,2))
        x_ = self.se(x_)
        # channel-wise multiplication
        x_ = x_.unsqueeze(-1).unsqueeze(-1)
        x = torch.mul(x,x_)
        return x"""


class SEBlock(nn.Module):
    #Paper: https://arxiv.org/pdf/1709.01507.pdf
    def __init__(self, block_in_channels, out_channels, reduction_ratio):
        super().__init__()
        self.block_in_channels = block_in_channels
        self.se_channels = out_channels
        self.r = reduction_ratio
        assert self.r > 0
        self.squeeze = int(self.block_in_channels * self.r)
        self.se = nn.Sequential(nn.Conv2d(self.se_channels, self.squeeze, kernel_size=1, bias=False),
                                nn.SiLU(),
                                nn.Conv2d(self.squeeze, self.se_channels, kernel_size=1, bias=False))

    def forward(self, x):
        x_se = torch.nn.functional.adaptive_avg_pool2d(x,1)        
        x_se = self.se(x_se)
        x = torch.sigmoid(x_se) * x
        return x


class MBConv(nn.Module):
    """Paper: https://arxiv.org/pdf/1801.04381.pdf"""
    def __init__(self, in_channels, out_channels, filter_size, expand_ratio, stride, se_ratio, bn_eps, bn_mom, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.se_ratio = se_ratio
        self.padding = padding
        self.expand_channels = self.in_channels * self.expand_ratio


        # Expansion 
        if self.expand_ratio != 1:
            self.expand = Conv2dSamePadding(in_channels=self.in_channels, out_channels=self.expand_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(num_features=self.expand_channels, momentum=bn_mom, eps=bn_eps)
        
        # Depthwise Convolution 
        self.d_conv = Conv2dSamePadding(in_channels=self.expand_channels, 
                                out_channels=self.expand_channels, 
                                groups=self.expand_channels, 
                                kernel_size=self.filter_size, 
                                stride=self.stride, 
                                bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.expand_channels, momentum=bn_mom, eps=bn_eps)

        # Squeeze and Excitation 
        if self.se_ratio:
            self.se = SEBlock(self.in_channels, self.expand_channels, se_ratio)

        # Pointwise (1x1) Convolution
        self.p_conv = Conv2dSamePadding(in_channels=self.expand_channels, out_channels=self.out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.out_channels, momentum=bn_mom, eps=bn_eps)
        self.swish = nn.SiLU()
    
    def forward(self, x):

        if self.expand_ratio != 1:
            x = self.expand(x)
            x = self.bn0(x)
            x = self.swish(x)
        
        x = self.d_conv(x)
        x = self.bn1(x)
        x = self.swish(x)

        if self.se_ratio:
            x = self.se(x)
        
        x = self.p_conv(x)
        x = self.bn2(x)

        # TODO 
        # implement skip connection and drop connect
        
        return x



class EfficientNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.num_classes = self.params.num_classes
        self.dropout = self.params.dropout
        self.in_channels = 3
        self.out_channels = int(1280 * self.params.width_coef)
        self.stages_config = self.params.stages
        # Batch norm parameters
        self.bn_mom = 1 - 0.99
        self.bn_eps = 1e-3

        self.stage1 = nn.Sequential(Conv2dSamePadding(self.in_channels, self.stages_config[0][3], 3, stride=2, bias=False),
                                    nn.BatchNorm2d(self.stages_config[0][3], eps=self.bn_eps, momentum=self.bn_mom),
                                    nn.SiLU(inplace=True))
        self.modules = [self._create_stage(params) for params in self.stages_config]
        self.stages = nn.Sequential(*self.modules)
        self.final = nn.Sequential(Conv2dSamePadding(self.stages_config[-1][4], self.out_channels, 1, bias=False),
                                   nn.BatchNorm2d(self.out_channels, eps=self.bn_eps, momentum=self.bn_mom),
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
        modules.append(MBConv(in_channels, out_channels, filter_size, expand_ratio, stride, se_ratio, 1, self.bn_eps, self.bn_mom))
        in_channels = out_channels
        if num_repeats > 1:
            for i in range(1, num_repeats):
                modules.append(MBConv(in_channels, out_channels, filter_size, expand_ratio, 1, se_ratio, 1, self.bn_eps, self.bn_mom))

        return nn.Sequential(*modules)


def test():
    # Test all EfficientNet configs
    effnet_versions = [
        'efficientnet-b0', 
        'efficientnet-b1',
        'efficientnet-b2',
        'efficientnet-b3',
        'efficientnet-b4',
        'efficientnet-b5',
        'efficientnet-b6',
        'efficientnet-b7',
    ] 
    
    for net in effnet_versions:
        print("Test: {}".format(net))
        params = ModelParams(net)
        model = EfficientNet(params)

        test_tensor = torch.randn(1,3,params.img_size,params.img_size)
        print("Input size: {}".format(test_tensor.size()))
        out = model(test_tensor)
        print("Output size: {}".format(out.size()))
        num_params = get_n_params(model)
        print("Number of parameters: {}".format(num_params))
        print("--"*20)


if __name__ == '__main__':
    test()