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
    def get_optimizer():
        return optim.SGD(params=filter(lambda p:p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4) #set optimizer
    
    def train_epoch():
        total_loss = []
        for img,target in tqdm(trainloader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)

            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            total_loss.append(loss.item())
            freeze_param(model)
            optimizer.step()

        return np.sum(total_loss)

    save_file_path = "log_file"
    os.makedirs(save_file_path, exist_ok=True)

    model = CNN(model_name="tiny",cls_num=10).train().to(device)
    eval_module = Evaluation()

    print("settings")
    trainset = torchvision.datasets.CIFAR10(root="./",train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=500,shuffle=True)

    epochs = args.epochs
    lr = args.lr
    
    print("epochs per iteration: [{}]".format(epochs))
    print("learning rate: [{}]".format(lr))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer()
    
    print("for late initialization")
    for img,target in tqdm(trainloader):
        optimizer.zero_grad()
        img = img.to(device)
        target = target.to(device)

        output = model(img)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print("late init")
    save_params(model)

    optimimzer = get_optimizer()
    EPOCH = []
    ACC = []
    LOSS = []
    """unpruned model"""
    for epoch in range(1, epochs+1, 1):
        print("epoch: [{}]/[{}]".format(epoch, epochs))
        if epoch in [0.5*epochs,0.8*epochs]:
            print("change learning rate")
            for param in optimizer.param_groups:
                param["lr"] *= 0.1

        total_loss = train_epoch()

        acc = eval_module.eval(model)

        EPOCH.append(epoch)
        ACC.append(acc)
        LOSS.append(total_loss)

    df = pd.DataFrame({
    "epoch": EPOCH,
    "accuracy": ACC,
    "loss":LOSS,
    })
    df.to_csv(save_file_path+"/unpruned_model.csv")

    """pruned_model"""
    prev_ratio = 10
    percent = prev_ratio/100.0
    model.prune(percent)
    load_params(model)
    model.mask_out() #masking
    prev_mask = model.mask
    ratio = [20, 30, 40, 50, 60, 70, 80, 90, 92, 95, 98, 99, 99]
    for r in ratio:
        optimizer = get_optimizer()
        
        EPOCH = []
        ACC = []
        LOSS = []
        for epoch in range(1, epochs+1, 1):
            print("epoch: [{}]/[{}]".format(epoch, epochs))
            if epoch in [0.5*epochs,0.8*epochs]:
                print("change learning rate")
                for param in optimizer.param_groups:
                    param["lr"] *= 0.1
            
            total_loss = train_epoch() #train per epoch

            acc = eval_module.eval(model)

            ACC.append(acc)
            EPOCH.append(epoch)
            LOSS.append(total_loss)

        df = pd.DataFrame({
        "epoch": EPOCH,
        "accuracy": ACC,
        "loss": LOSS,
        })
        df.to_csv(save_file_path+"/pruned_"+str(int(prev_ratio))+"_percent.csv")

        percent = (r-prev_ratio)/100.0
        model.prune(percent) #prune model
        mask = model.mask #next mask

        print("in case of random initalization with the same network")
        model = CNN(model_name="tiny",cls_num=10).train().to(device)
        model.mask = prev_mask
        model.mask_out() #masking
        
        optimizer = get_optimizer()

        EPOCH = []
        ACC = []
        LOSS = []
        for epoch in range(1, epochs+1, 1):
            print("epoch: [{}]/[{}]".format(epoch, epochs))
            if epoch in [0.5*epochs,0.8*epochs]:
                print("change learning rate")
                for param in optimizer.param_groups:
                    param["lr"] *= 0.1
            
            total_loss = train_epoch() #train per epoch

            acc = eval_module.eval(model)

            ACC.append(acc)
            EPOCH.append(epoch)
            LOSS.append(total_loss)

        df = pd.DataFrame({
        "epoch": EPOCH,
        "accuracy": ACC,
        "loss": LOSS,
        })
        df.to_csv(save_file_path+"/pruned_"+str(int(prev_ratio))+"_percent_with_random_init.csv")

        load_params(model) #load
        model.mask = mask
        model.mask_out()
        prev_mask = mask
        prev_ratio = r

if __name__ == "__main__":
    train()
