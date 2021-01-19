#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn

class SwitchableConv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, without_input=False):
        super(SwitchableConv2d, self).__init__(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=1,
        groups=groups, bias=bias)
        self.in_size = in_channel
        self.out_size = out_channel
        self.without_input = without_input


    def forward(self, x, width):
        if self.without_input:
            self.in_channels = self.in_size
        else:
            self.in_channels = int(self.in_size*width)

        self.out_channels = int(self.out_size*width)

        weight = self.weight[:self.out_channels,:self.in_channels]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = None

        y = nn.functional.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        return y

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, size, widths):
        super(SwitchableBatchNorm2d, self).__init__()
        s_bn = []
        self.width2idx = {}
        for width in widths:
            self.width2idx[width] = len(s_bn)
            s_bn.append(nn.BatchNorm2d(int(size*width)))

        self.s_bn = nn.Sequential(*s_bn)

    def forward(self, x, width):
        idx = self.width2idx[width]

        return self.s_bn[idx](x)

class SwitchableLinear(nn.Module):
    def __init__(self, in_size, out_size, without_output=False):
        super(SwitchableLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.randn(out_size, in_size))
        self.bias = nn.Parameter(torch.randn(out_size))
        self.without_output = without_output

    def forward(self, x, width):
        in_size = int(self.in_size*width)
        if not self.without_output:
            out_size = int(self.out_size*width)
        else:
            out_size = self.out_size

        weight = self.weight[:out_size, :in_size]
        bias = self.bias[:out_size]

        return torch.matmul(x, weight.t()) + bias

class SwitchableModel(nn.Module):
    def __init__(self, switchable_list):
        super(SwitchableModel, self).__init__()
        self.models = self.create_layer(switchable_list)
       
        self.classifier = nn.Sequential(
        SwitchableLinear(512, 10, without_output=True),
        )

    def forward(self, x, width):
        for layer in self.models:
            if isinstance(layer, SwitchableConv2d) or isinstance(layer, SwitchableBatchNorm2d):
                x = layer(x, width)
            else:
                x = layer(x)

        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            if isinstance(layer, SwitchableLinear):
                x = layer(x, width)
            else:
                x = layer(x)

        return x

    def create_layer(self, widths):
        layer_list = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]
        layers = []
        inc = 3
        flag = True
        for outc in layer_list:
            if outc is not "M":
                layers.append(SwitchableConv2d(inc, outc, 3, 1, 1, without_input=flag))
                layers.append(SwitchableBatchNorm2d(outc, widths))
                layers.append(nn.ReLU())
                inc = outc
                flag = False
            else:
                layers.append(nn.MaxPool2d(2,2))

        layers.append(nn.AdaptiveAvgPool2d(1))
        return nn.Sequential(*layers)
                
