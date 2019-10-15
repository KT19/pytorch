#-*-coding:utf-8-*-
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from config import *

class ArcFace(nn.Module):
    def __init__(self, n_class=10):
        super(ArcFace, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=(4,4),stride=(2,2),padding=(1,1)), #14
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=(4,4),stride=(2,2),padding=(1,1)), #7
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(0, 0)), #5
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(0, 0)), #3
        nn.BatchNorm2d(128),
        nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.metric = ArcFace_Layer(128, n_class)

    def forward(self, x, gt=None):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        return self.metric(x, gt)

class ArcFace_Layer(nn.Module):
    def __init__(self,in_size, out_size,s=64,m=0.5):
        super(ArcFace_Layer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.scale = s
        self.margin = m
        self.w = Parameter(torch.FloatTensor(out_size,in_size))
        nn.init.xavier_uniform_(self.w)

    def forward(self, x, gt):
        """
        args:
        x: feature vector(N, D)
        gt: labels(optional, only when training)
        """
        if gt is None:
            return self.scale*F.linear(F.normalize(x),F.normalize(self.w))

        original_target_logit = []
        logit = F.linear(F.normalize(x),F.normalize(self.w))
        for i,label in enumerate(gt): #pick up target logit
            original_target_logit.append(logit[i,label].item())
        original_target_logit = torch.Tensor(original_target_logit).view(-1,1)

        theta = torch.acos(original_target_logit)
        marginal_target_logit = torch.cos(theta+self.margin)
        one_hot = torch.Tensor(np.eye(self.out_size)[gt]).to(device)

        output = logit + one_hot*(marginal_target_logit - original_target_logit)
        return self.scale * output
