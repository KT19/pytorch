#-*-coding:utf-8-*-
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from configure import *

models = {
"tiny": [32, "D", 64, "D", 128, "D", 256, "D", 512],
"vgg16": [64, 64, "D", 128, 128, "D", 256, 256, 256, "D", 512, 512, 512, "D", 512, 512, 512],
}

class Model(nn.Module):
    def __init__(self, model_name, cls_num):
        super(Model, self).__init__()

        self.features = create_module(models[model_name])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, cls_num)

    def forward(self, x):
        y = self.features(x)
        y = self.gap(y)
        y = y.view(-1, 512)
        return self.fc(y)

def create_module(lis,batchnorm=True):
    layers = []
    inc = 3
    for l in lis:
        if l is not "D":
            conv = nn.Conv2d(inc, l, kernel_size=3, padding=1)
            if batchnorm:
                layers += [conv, nn.BatchNorm2d(l), nn.ReLU()]
            else:
                layers += [conv, nn.ReLU()]
            inc = l
        else:
            layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]

    return nn.Sequential(*layers)
