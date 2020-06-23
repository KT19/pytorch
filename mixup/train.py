#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
from modules import *
from configure import *
from eval import Eval

def train():
    trainset = torchvision.datasets.CIFAR10(root="~/dataset",train=True,transform=transform,download=True)
    trainloader = torch.utils.data.DataLoader(trainset, args.batch_size, shuffle=True)

    evaluator = Eval()
    model = Model("vgg16",cls_num=10)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.wd,momentum=args.momentum)
    
    if args.mode == "normal":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = MixUpLoss()
    n_epochs = args.n_epochs

    LOSS_LOG = []
    ACC_LOG = []
    EPOCHS = []
    for epoch in range(1,n_epochs+1,1):
        print("Epoch:[{}]/[{}]".format(epoch,n_epochs))
        if epoch in [0.5*n_epochs, 0.8*n_epochs]:
            print("change learning rate")
            for param in optimizer.param_groups:
                param["lr"] *= 0.1

        total_loss = []
        for img, label in tqdm(trainloader):
            optimizer.zero_grad()

            img = img.to(device)
            label = label.to(device)

            #mixup
            if args.mode == "mixup":
                img,y1,y2,lam = create_mixup(img, label, alpha=args.alpha)

            #forward
            pred = model(img)
            #backward
            if args.mode == "normal": #normal loss
                loss = criterion(pred,label)
            else: #mixup loss
                loss = criterion(pred, y1, y2, lam)
            
            loss.backward()

            optimizer.step()
            total_loss.append(loss.item())

        acc = evaluator.eval(model)
        mean_loss = np.mean(total_loss)
        print("loss: {}".format(mean_loss))
        print("accuracy: {}[%]".format(acc))

        LOSS_LOG.append(mean_loss)
        ACC_LOG.append(acc)
        EPOCHS.append(epoch)
        df = pd.DataFrame({
        "LOSS":LOSS_LOG,
        "ACCURACY": ACC_LOG,
        "EPOCH": EPOCHS,
        })
        df.to_csv("./"+args.mode+"_train_log.csv")


if __name__=="__main__":
    train()
