#-*-coding:utf-8-*-
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
from configure import *

models = {
"tiny":[32, "D", 64, "D", 128],
}
class CNN(nn.Module):
    def __init__(self, model_name="tiny", cls_num=10):
        super(CNN, self).__init__()
        self.model = self.create_model(model_name)
        self.fc = nn.Linear(models[model_name][-1], cls_num)

        self.create_mask() #for winning ticket

    def create_model(self, model_name):
        prev_c = 3
        model_list = models[model_name]
        modules = []
        for l in model_list:
            if l == "D":
                modules.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
            else:
                modules.append(nn.Conv2d(prev_c, l, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
                modules.append(nn.ReLU())
                prev_c = l

        modules.append(nn.AdaptiveAvgPool2d(1))

        return nn.Sequential(*modules)

    def create_mask(self):
        self.mask = []
        self.model_params = []

        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                o_size,i_size,kh,kw = layer.weight.size()
                self.mask.append(torch.ones(o_size, i_size, kh, kw, device=device))
                self.model_params.append(o_size*i_size*kh*kw)

        self.prev_masks = [0 for i in range(len(self.mask))]

    def forward(self, x):
        y = self.model(x)
        y = y.view(y.size(0), -1)
        return self.fc(y)

    def prune(self, rat):
        print("select weights which are pruned")
        print(rat)
        curr = 0

        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
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
                o_s,i_s,kh,kw =  mask.size()
                mask = mask.view(-1)
                mask[indicies] = 0
                mask = mask.view(o_s, i_s, kh, kw)
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
            if isinstance(layer, nn.Conv2d):
                layer.weight = nn.Parameter(layer.weight.data*self.mask[curr])
                curr += 1

class Evaluation():
    def __init__(self):
        print("set loader")
        testset = torchvision.datasets.CIFAR10(root="./",train=False,download=True,transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)

    def eval(self, model):
        model.eval()
        total = 0
        acc = 0

        with torch.no_grad():
            for img,target in tqdm(self.testloader):
                img = img.to(device)
                total += img.size(0)

                output = model(img)
                pred = torch.argmax(output, 1)

                acc += (pred == target).sum().item()

        model.train()
        print("accuracy is {}".format((acc/total)*100))

        return (acc/total)*100
