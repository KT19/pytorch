#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=(4,4), stride=(2,2), padding=(1,1)), #32
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1)), #16
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),
        nn.Conv2d(256, 512, kernel_size=(4,4), stride=(2,2), padding=(1,1)), #8
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2),
        nn.Conv2d(512, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1)), #4
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),
        nn.Conv2d(256, ENC_LATENT, kernel_size=(4,4), stride=(2,2), padding=(1,1)), #2
        nn.BatchNorm2d(ENC_LATENT),
        nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        Ex = self.model(x)
        return Ex

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
        nn.ConvTranspose2d(ENC_LATENT + ATTR, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1)), #4
        nn.BatchNorm2d(256),
        nn.ReLU()
        )
        self.layer2 = nn.Sequential(
        nn.ConvTranspose2d(256 + ATTR, 512, kernel_size=(4,4), stride=(2,2),padding=(1,1)), #8
        nn.BatchNorm2d(512),
        nn.ReLU()
        )
        self.layer3 = nn.Sequential(
        nn.ConvTranspose2d(512 + ATTR, 256, kernel_size=(4,4), stride=(2,2),padding=(1,1)), #16
        nn.BatchNorm2d(256),
        nn.ReLU()
        )
        self.layer4 = nn.Sequential(
        nn.ConvTranspose2d(256 + ATTR, 128, kernel_size=(4,4), stride=(2,2),padding=(1,1)), #32
        nn.BatchNorm2d(128),
        nn.ReLU()
        )
        self.layer5 = nn.Sequential(
        nn.ConvTranspose2d(128 + ATTR, 3, kernel_size=(4,4), stride=(2,2),padding=(1,1)), #64
        nn.BatchNorm2d(3),
        nn.ReLU()
        )

    def forward(self, Ex, y):
        """
        layer1
        """
        Ex = Ex.view((Ex.size(0), -1, 2, 2))
        attr = torch.zeros((Ex.size(0), ATTR, 2, 2)).to(device)
        attr[y == 1] = 1.0
        latent = torch.cat((Ex, attr),1)
        Ex = self.layer1(latent)
        """
        layer2
        """
        attr = torch.zeros((Ex.size(0), ATTR, 4, 4)).to(device)
        attr[y == 1] = 1.0
        latent = torch.cat((Ex, attr),1)
        Ex = self.layer2(latent)
        """
        layer3
        """
        attr = torch.zeros((Ex.size(0), ATTR, 8, 8)).to(device)
        attr[y == 1] = 1.0
        latent = torch.cat((Ex, attr),1)
        Ex = self.layer3(latent)
        """
        layer4
        """
        attr = torch.zeros((Ex.size(0), ATTR, 16, 16)).to(device)
        attr[y == 1] = 1.0
        latent = torch.cat((Ex, attr),1)
        Ex = self.layer4(latent)
        """
        layer5
        """
        attr = torch.zeros((Ex.size(0), ATTR, 32, 32)).to(device)
        attr[y == 1] = 1.0
        latent = torch.cat((Ex, attr),1)
        generated = self.layer5(latent)

        return generated

class Fader_Network(nn.Module):
    def __init__(self):
        super(Fader_Network, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, y):
        return self.decoder(x, y)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
        nn.Linear(ENC_LATENT*4, 256),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(256, ATTR),
        nn.Sigmoid(),
        )

    def forward(self, Ex):
        Ex = Ex.view(Ex.size(0),-1)
        return self.model(Ex)
