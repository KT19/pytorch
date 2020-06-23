#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np

class MixUpLoss(nn.Module):
    def __init__(self):
        super(MixUpLoss,self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, y1, y2, lam):
        """
        args:
        output: output of model
        y1: label1
        y2: label2
        lam: hyper parameter of lamba in mixpu
        """
        return lam*self.loss(output,y1)+(1.0-lam)*self.loss(output,y2)

def create_mixup(x,y,alpha=0.2):
    """
    args:
    x: datas
    y: one-hot labels
    alpha(optional): hyper parameters
    """
    lam = np.random.beta(alpha, alpha) #get ramdom value
    N = x.size(0) #batch size
    rand_idx = np.array([i for i in range(N)]) #create index
    np.random.shuffle(rand_idx)

    #get random shuffle
    x2 = x[rand_idx]
    y2 = y[rand_idx]
    #mixup
    mix = lam*x+(1.0-lam)*x2

    return mix, y, y2, lam
