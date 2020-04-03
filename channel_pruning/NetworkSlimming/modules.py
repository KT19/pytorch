#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from config import *

models = {
"tiny": [16, "D", 32, "D", 64],
"vgg16": [64, 64, "D", 128, 128, "D", 256, 256, 256, "D", 512, 512, 512, "D", 512, 512, 512]
}

class CNN(nn.Module):
    def __init__(self, model):
        super(CNN, self).__init__()
        self.model = self.create_model(models[model])
        self.gap = nn.AdaptiveAvgPool2d(1)

        in_size = models[model][-1]
        self.fc = nn.Linear(in_size, 10)

    def create_model(self, layers):
        prev_channel = 3
        modules = []
        for l in layers:
            if l == "D":
                modules.append(nn.MaxPool2d(2, 2))
            else:
                modules.append(nn.Conv2d(prev_channel, l, kernel_size=(3,3), stride=(1, 1), padding=(1, 1)))
                modules.append(nn.BatchNorm2d(l))
                modules.append(nn.ReLU())
                prev_channel = l
        return nn.Sequential(*modules)

    def forward(self, x):
        y = self.model(x)
        y = self.gap(y)
        y = y.view(y.size(0), -1)
        return self.fc(y)

class Evaluation:
    def __init__(self):
        testset = torchvision.datasets.CIFAR10(root="./",train=True,download=True,transform=testtransform)
        self.testloader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size, shuffle=True,drop_last=True)

    def eval(self, model):
        model.eval()
        total = 0
        ans = 0
        with torch.no_grad():
            for img,target in tqdm(self.testloader):
                total += img.size(0)
                output = model(img)
                pred = torch.argmax(output, 1)

                ans += (pred == target).sum().item()

        acc = (ans/total)*100
        model.train()
        print("accuracy :[{}]".format(acc))

        return acc
