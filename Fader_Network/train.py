#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import *
from utils import *
from module import *
import matplotlib.pyplot as plt

def Adversarial_Train(x, y, fader, discriminator, fader_optimizer, discriminator_optimizer, le=0, clip=10):
    def reset_grad():
        fader_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()


    """
    discriminator objective
    """
    attribute = y.to(device)
    reset_grad()
    Ex = fader.encode(x)
    Py = discriminator(Ex.detach())
    loss = F.binary_cross_entropy(Py, attribute,reduction="sum")
    loss.backward()
    #update
    discriminator_optimizer.step()

    """
    Encoder-Decoder loss
    """
    reset_grad()
    Ex = fader.encode(x)
    Py = discriminator(Ex)
    loss_ae = F.mse_loss(fader.decode(fader.encode(x), y) , x)

    """
    Adversarial objective
    """
    Loss = loss_ae - le*F.binary_cross_entropy(Py, attribute, reduction="sum")
    Loss.backward()
    fader_optimizer.step()

    return Loss.item()

def eval():
    def visualize(x, y, fader, name="manipulating img"):
        with torch.no_grad():
            outputs = fader.decode(fader.encode(x), y)
            outputs = outputs.squeeze(0)
            outputs = outputs.to("cpu").numpy().transpose(1,2,0)
            plt.imshow(outputs*0.5 + 0.5)
            plt.title(name)
            plt.show()

        return

    #model set
    fader = Fader_Network()
    fader.load_state_dict(torch.load("fader"))
    fader = fader.to(device)
    fader.eval()

    #dataloader
    testloader,attr2idx = get_loader(image_dir="./data/img_celebA", attr_path="./data/list_attr_celeba.txt",batch_size=1,mode="test")

    print(attr2idx)
    while True:
        (x, y) = next(iter(testloader))
        x = x.to(device)
        y = torch.FloatTensor(y).to(device)
        visualize(x, y, fader)
        while True:
            print("input label '7 1 5 ・・・' enter is finished")
            str = input()
            str = str.split()
            y = torch.zeros(1, ATTR)
            for v in str:
                label = int(v)
                y[0,label] = 1.

            y = y.to(device)
            visualize(x, y, fader)

def train():
    def visualize(x, y, fader):
        outputs = fader.decode(fader.encode(x), y)
        torchvision.utils.save_image(0.5*outputs+0.5, "faded.png", nrow=10)
    #model set
    fader = Fader_Network()
    fader = fader.to(device)
    fader.train()
    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    discriminator.train()

    #dataloader
    trainloader,_ = get_loader(image_dir="./data/img_celebA", attr_path="./data/list_attr_celeba.txt", batch_size=100, mode="train", num_workers=1)
    fader_optimizer = optim.Adam(fader.parameters(),lr=1e-3)
    discriminator_optimizer = optim.Adam(discriminator.parameters(),lr=1e-3)
    h = 0.0001
    le = 0.0

    n_iter = 10000

    for it in tqdm(range(1, n_iter, 1)):
        (x, y) = next(iter(trainloader))

        x = x.to(device)
        #y = convert_ann2y(annotation)
        #y = [0, 1, 1, 0, ・・・](対応するラベルが1)であればよい

        loss = Adversarial_Train(x, y, fader, discriminator, fader_optimizer, discriminator_optimizer, le)
        le = le + h

        if(it %500 == 0):
            torch.save(fader.state_dict(),"fader")
            torch.save(discriminator.state_dict(),"discriminator")
            visualize(x, y, fader)
            print("loss :{}".format(loss))
