#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from module import *
from config import *

test_batch_size = 10
testset = torchvision.datasets.MNIST(root="./",train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=test_batch_size)

def eval():
    print("check accuracy")
    model = ArcFace()
    model = model.eval().to(device)
    model.load_state_dict(torch.load("./arcface.pth"))

    total = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in tqdm(testloader):
            data = data.to(device)
            target = target.to(device)

            pred = model(data)
            _, pred = torch.max(pred.data,1)
            total += pred.size(0)
            correct += (pred == target).sum().item()

    acc = (correct / total)*100.
    print("accuracy is {}".format(acc))

    return acc
