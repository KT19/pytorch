#-*-coding:utf-8-*-
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from module import *
from eval import *
from config import *

def train():
    #set model
    model = ArcFace()
    model.train().to(device)

    #dataset
    batch_size = 32
    trainset = torchvision.datasets.MNIST(root="./",train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32, shuffle=True,drop_last=True)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    n_epoch = 5
    for epoch in range(1,n_epoch+1,1):
        print("Epoch:[{}] / [{}]".format(epoch,n_epoch))
        total_loss = []
        for (data, target) in tqdm(trainloader):
            data = data.to(device)
            target = target.to(device)

            y = model(data, target)
            loss = criterion(y, target)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())
        print("loss: [{}]".format(np.mean(np.array(total_loss))))
        torch.save(model.state_dict(),"./arcface.pth")
        eval() #evaluation

if __name__ == "__main__":
    train()
