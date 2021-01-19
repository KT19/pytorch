#-*-coding:utf-8-*-
import numpy as np
import torch
import torchvision
import pandas as pd
from tqdm import tqdm
from modules import SwitchableModel
from eval import eval
from config import transform, testtransform, device

def train():
    epochs = 100
    switchable_list = [0.25, 0.5, 0.75, 1.0]
    model = SwitchableModel(switchable_list)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    trainset = torchvision.datasets.CIFAR10("~/dataset",train=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

    testset = torchvision.datasets.CIFAR10("~/dataset",train=False,transform=testtransform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    LOGS = {}
    LOGS["epoch"] = []
    for width in switchable_list:
        LOGS[width] = []

    for epoch in range(1, epochs+1, 1):
        model.train()
        print("Epoch:[{}]/[{}]".format(epoch, epochs))
        epoch_loss = []
        for img, target in tqdm(trainloader):
            img = img.to(device)
            target = target.to(device)

            switch_loss = []
            for width in switchable_list:
                optimizer.zero_grad()
                def closure():
                    out = model(img, width)
                    loss = criterion(out, target)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                switch_loss.append(loss.item())
            epoch_loss.append(np.mean(switch_loss))

        accs = eval(model, testloader, switchable_list)
        print("Loss: [{}]".format(np.mean(epoch_loss)))

        for width in switchable_list:
            print("width = [{}], Acc: [{}][%]".format(width, accs[width]))
            LOGS[width].append(accs[width])
        LOGS["epoch"].append(epoch)

        scheduler.step()
        df = pd.DataFrame(LOGS)
        df.to_csv("log_state.csv")


if __name__ =="__main__":
    train()
