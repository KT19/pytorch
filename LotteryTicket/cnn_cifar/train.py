#-*-coding:utf-8-*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from modules import *
from configure import *
from plot import *

def save_params(model,path="save_model"):
    """
    args:
    model: model
    path: path to save directory
    """
    model = model.to("cpu")
    model_name = "cnn_model.pth"
    os.makedirs(path,exist_ok=True)
    torch.save(model.state_dict(),path+"/"+model_name)
    print("save model parameters")
    model = model.to(device)

def load_params(model, path="save_model"):
    model = model.to("cpu")
    model_name = "cnn_model.pth"
    model.load_state_dict(torch.load(path+"/"+model_name))
    print("load model parameters")
    model = model.to(device)

def freeze_param(model):
    """
    freezing parameters
    """
    for name,param in model.named_parameters():
        if "weight" in name:
            grad = param.grad.data.to("cpu").numpy()
            masked_grad = np.where(param.to("cpu") == 0, 0, grad)
            param.grad.data = torch.Tensor(masked_grad).to(device)

def train():
    save_file_path = "log_file"
    os.makedirs(save_file_path, exist_ok=True)

    model = CNN().train().to(device)
    eval_module = Evaluation()

    print("save initial model params")
    save_params(model)

    print("settings")
    trainset = torchvision.datasets.CIFAR10(root="./",train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=100,shuffle=True)

    iteration = args.iteration+1
    epochs = args.epochs
    lr = args.lr
    pruned_ratio = args.ratio / args.iteration #per

    print("iteration (how many times the network is pruned): [{}]".format(args.iteration))
    print("epochs per iteration: [{}]".format(epochs))
    print("learning rate: [{}]".format(lr))
    print("pruned ratio per iteration: [{}]".format(pruned_ratio))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=filter(lambda p:p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)

    for itr in range(1, iteration+1, 1):
        print("iteration: [{}]/[{}]".format(itr, iteration))
        EPOCH = []
        ACC = []
        for epoch in range(1, epochs+1, 1):
            print("epoch: [{}]/[{}]".format(epoch, epochs))
            for img,target in tqdm(trainloader):
                break
                optimizer.zero_grad()
                img = img.to(device)
                target = target.to(device)

                output = model(img)
                loss = criterion(output, target)
                loss.backward()
                freeze_param(model)
                optimizer.step()

            acc = eval_module.eval(model)

            ACC.append(acc)
            EPOCH.append(epoch)
        df = pd.DataFrame({
        "epoch": EPOCH,
        "accuracy": ACC,
        })
        df.to_csv(save_file_path+"/iteration_"+str(itr)+".csv")

        if itr is not iteration:
            model.prune(pruned_ratio) #prune model
            load_params(model)
            model.mask_out() #masking
            optimizer = optim.SGD(params=filter(lambda p:p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4) #new optimizer

    mask = model.mask
    print("in case of random initial weight with the same network")
    model= CNN().train().to(device)
    model.mask = mask
    model.mask_out() #masking
    optimizer = optim.SGD(params=filter(lambda p:p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4) #new optimizer

    ACC = []
    EPOCH = []
    for epoch in range(1, epochs+1, 1):
        print("epoch: [{}]/[{}]".format(epoch, epochs))
        for img,target in tqdm(trainloader):
            break
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)

            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            freeze_param(model)
            optimizer.step()

        acc = eval_module.eval(model)

        ACC.append(acc)
        EPOCH.append(epoch)
    df = pd.DataFrame({
    "epoch": EPOCH,
    "accuracy": ACC,
    })
    df.to_csv(save_file_path+"/random_init.csv")

    plot_results()


if __name__ == "__main__":
    train()
