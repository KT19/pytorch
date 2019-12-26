#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from configure import *

class SubGenerator(nn.Module):
    def __init__(self, channel=32):
        super(SubGenerator, self).__init__()

        self.inc = 3

        self.model = nn.Sequential(
        nn.Conv2d(self.inc, channel, 3, 1),
        nn.BatchNorm2d(channel),
        nn.LeakyReLU(0.2),
        nn.Conv2d(channel, channel, 3, 1),
        nn.BatchNorm2d(channel),
        nn.LeakyReLU(0.2),
        nn.Conv2d(channel, channel, 3, 1),
        nn.BatchNorm2d(channel),
        nn.LeakyReLU(0.2),
        nn.Conv2d(channel, channel, 3, 1),
        nn.BatchNorm2d(channel),
        nn.LeakyReLU(0.2),
        nn.Conv2d(channel, 3, 3, 1),
        nn.Tanh(),
        )

    def forward(self, noise, y):
        """
        noise: noise
        y: previous outputs
        """
        noise = F.pad(noise,[5,5,5,5], value=0)

        if y is None:
            o = self.model(noise)
        else:
            padded_y = F.pad(y,[5,5,5,5], value=0)
            o = self.model(noise+padded_y) + y

        return o

class Generator(nn.Module):
    def __init__(self,model_num,inc=32,scale_up_step=4,scale=4./3.):
        super(Generator, self).__init__()

        self.size_list = [int(IMG_SIZE/(scale**i)) for i in range(model_num)] #each outputs of image
        self.size_list.reverse() #descending order
        self.img_list = []
        print("size is {}".format(self.size_list))

        self.model_num = model_num
        self.current_scale_num = 1
        self.scale_up_step = scale_up_step
        self.scale = scale

        model_list = nn.ModuleList()

        #create models
        for i in range(1,model_num+1,1):
            if i % scale_up_step == 0:
                inc = int(inc*2)

            model_list.append(SubGenerator(channel=inc))

        self.models = nn.Sequential(*model_list)

    def set_img_list(self, img):
        """
        args:
        original img
        """
        for s in self.size_list:
            self.img_list.append(F.interpolate(img,(s,s)))

    def forward(self, z, eval=False):
        """
        args:
        z: input noise or zeros
        eval(optional): for evaluation
        """
        outputs = []
        for i in range(self.current_scale_num):
            if i == 0:
                xhat = self.models[i](z[i], None) #bottom models
            else:
                if eval:
                    xhat = F.interpolate(xhat,scale_factor=self.scale,mode="bilinear",align_corners=True)
                    xhat = self.models[i](z[i], xhat)

                else:
                    size = self.size_list[i]
                    xhat = F.interpolate(xhat,size=(size,size),mode="bilinear",align_corners=True)
                    xhat = self.models[i](z[i], xhat)

            outputs.append(xhat)

        return outputs

    def scale_up(self):
        for param in self.models[self.current_scale_num-1].parameters():
            param.requires_grad = False
        self.models[self.current_scale_num-1].eval()

        self.current_scale_num += 1
        print("scale up to {} in generator".format(self.current_scale_num))

        #initialize
        if self.current_scale_num % self.scale_up_step != 0:
            print("generator is initialized via weight copy")
            self.models[self.current_scale_num-1].load_state_dict(self.models[self.current_scale_num-2].state_dict())

class SubDiscriminator(nn.Module):
    def __init__(self,inc):
        super(SubDiscriminator,self).__init__()
        self.inc = inc

        self.layers = nn.Sequential(
        nn.Conv2d(3, inc, 3, 1),
        nn.BatchNorm2d(inc),
        nn.LeakyReLU(0.2),
        nn.Conv2d(inc, inc, 3, 1),
        nn.BatchNorm2d(inc),
        nn.LeakyReLU(0.2),
        nn.Conv2d(inc, inc, 3, 1),
        nn.BatchNorm2d(inc),
        nn.LeakyReLU(0.2),
        nn.Conv2d(inc, inc, 3, 1),
        nn.BatchNorm2d(inc),
        nn.LeakyReLU(0.2),
        nn.Conv2d(inc, 1, 3, 1),
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self,inc=32,scale_up_step=4):
        super(Discriminator,self).__init__()

        self.scale_num = 1
        self.inc = inc
        self.scale_up_step = scale_up_step

        self.model = SubDiscriminator(inc)

    def forward(self, x):
        """
        input:
        x (B,3,H,W)
        """
        y = self.model(x)

        return y

    def scale_up(self):
        print("scale up in discriminator")
        self.scale_num += 1
        if self.scale_num % self.scale_up_step == 0:
            self.inc = int(2*self.inc)
            self.model = SubDiscriminator(self.inc)
            self.model.to(device)
        else:
            print("discriminator is initialized via weight copy")
            prev_state = self.model.state_dict()
            self.model = SubDiscriminator(self.inc)
            self.model.to(device)
            self.model.load_state_dict(prev_state)

if __name__ =="__main__":
    generator = Generator()
    discriminator = Discriminator()

    x = torch.randn(8, 3, 256, 256)
    outputs = generator(x)

    x5,x4,x3,x2,x1 = outputs
    outputs = discriminator(x1,x2,x3,x4,x5)

    for o in outputs:
        print(o.shape)
