#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import pandas as pd
from tqdm import tqdm
from utils import *
from modules import *
from config import *

def count_params(model):
    param = 0
    for p in model.parameters():
        if p.requires_grad:
            param += p.numel()

    return param

def train():
    def add_sparsity(model):
        """
        add sparsity conjuction with BatchNorm
        """
        g_s = 0
        for layer in model.model:
            if isinstance(layer, nn.BatchNorm2d):
                g_s += torch.sum(abs(layer.weight))

        return g_s

    print("train with network slimming")
    model = CNN(model="tiny")
    eval = Evaluation()

    optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    batch_size = args.batch_size
    epochs = args.epochs

    trainset = torchvision.datasets.CIFAR10(root="./",train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True,drop_last=True)

    criterion = nn.CrossEntropyLoss()

    print("learning rate: [{}]".format(args.lr))
    print("momentum: [{}]".format(args.momentum))
    print("weight_decay: [{}]".format(args.wd))
    print("sparsity: [{}]".format(args.sparsity))
    print("prune ratio at one time: [{}]".format(args.ratio))

    EPOCHS = []
    ACC = []
    LOSS = []
    PARAMS = []
    for epoch in range(1, epochs+1, 1):
        if epoch in [0.5*epochs]:
            prune_model(model, args.ratio) #pruning
            print(model)

        total_loss = []
        print("Epoch: [{}]/[{}]".format(epoch, epochs))
        for img,target in tqdm(trainloader):
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, target)
            loss += args.sparsity*add_sparsity(model)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        total_loss = np.mean(total_loss)
        print("loss: [{}]".format(total_loss))
        acc = eval.eval(model)
        EPOCHS.append(epoch)
        ACC.append(acc)
        LOSS.append(total_loss)
        PARAMS.append(count_params(model))

        df = pd.DataFrame({
        "accuracy": ACC,
        "loss": LOSS,
        "params": PARAMS,
        "epoch": EPOCHS
        })
        df.to_csv("train_log.csv")


if __name__ == "__main__":
    train()
