#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import pandas as pd
from tqdm import tqdm
from modules import *
from config import *

def prune_model(model, ratio):
    """
    args:
    model: cnn model with batch norm
    ratio: the ratio of how many parameters should be pruned
    """
    print("pruning each layer about {}[%]".format(ratio))

    flag = False #used to check next layer
    for layer in model.model:
        if isinstance(layer, nn.Conv2d):
            if flag: #prune next channel
                flag = False
                layer.in_channels = len(indicies)
                layer.weight = nn.Parameter(layer.weight[:,indicies])

            prev_layer = layer

        if isinstance(layer, nn.BatchNorm2d):
            """
            frist sorting based on magnitude
            """
            params = layer.num_features
            pruned_params = int(ratio*params)
            weight = abs(layer.weight)
            indicies = torch.argsort(weight)[pruned_params:]
            """
            prune params in batch norm
            """
            layer.weight = nn.Parameter(layer.weight[indicies])
            layer.bias = nn.Parameter(layer.bias[indicies])
            layer.running_mean = layer.running_mean[indicies]
            layer.running_var = layer.running_var[indicies]
            layer.num_features = len(indicies)

            """
            prune params in previous conv layer
            """
            prev_layer.weight = nn.Parameter(prev_layer.weight[indicies])
            prev_layer.bias = nn.Parameter(prev_layer.bias[indicies])
            prev_layer.out_channels = len(indicies)

            flag = True #used to prune next convolution filter

    """
    prune fc layer
    """
    model.fc.weight = nn.Parameter(model.fc.weight[:,indicies])
    model.fc.in_features = len(indicies)
