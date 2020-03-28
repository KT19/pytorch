#-*-coding:utf-8-*-
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from configure import *

class MLP(nn.Module):
    def __init__(self, hidden=128):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(28*28, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, 10, bias=False)
        )

        self.create_mask() #for winning ticket

    def create_mask(self):
        self.mask = []
        self.model_params = []

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                row,col = layer.weight.size()
                self.mask.append(torch.ones(row, col))
                self.model_params.append(row*col)

        self.prev_masks = [0 for i in range(len(self.mask))]

    def forward(self, x):
        return self.model(x)

    def prune(self, rat):
        print("select weights which are pruned")
        print(rat)
        curr = 0

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                """
                sort according to magnitude
                """
                prune_params = int(self.model_params[curr]*rat)
                weight = layer.weight.data.reshape(-1)
                weight = abs(weight)
                indicies = torch.argsort(weight)[self.prev_masks[curr]:self.prev_masks[curr]+prune_params] #for array

                """
                zero out
                """
                mask = self.mask[curr]
                row,col =  mask.size()
                mask = mask.view(-1)
                mask[indicies] = 0
                mask = mask.view(row, col)
                self.mask[curr] = mask

                one_num = mask[mask == 1].sum()
                print(prune_params)
                print(one_num)

                self.prev_masks[curr] += prune_params
                curr += 1

    def mask_out(self):
        print("mask (zero and fixed params)")
        curr = 0
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                layer.weight = nn.Parameter(layer.weight.data*self.mask[curr])
                curr += 1

class Evaluation():
    def __init__(self):
        print("set loader")
        testset = torchvision.datasets.MNIST(root="./",train=False,download=True,transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)

    def eval(self, model):
        model.eval()
        total = 0
        acc = 0

        with torch.no_grad():
            for img,target in tqdm(self.testloader):
                img = img.view(img.size(0),-1)
                total += img.size(0)

                output = model(img)
                pred = torch.argmax(output, 1)

                acc += (pred == target).sum().item()

        model.train()
        print("accuracy is {}".format((acc/total)*100))

        return (acc/total)*100
