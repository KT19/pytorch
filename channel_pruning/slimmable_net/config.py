#-*-coding:utf-8-*-
import torch
import torchvision.transforms as T

testtransform = T.Compose([
T.ToTensor(),
T.Normalize((0.5,0.5, 0.5),(0.5, 0.5, 0.5))
])

transform = T.Compose([
T.RandomCrop((32,32),padding=4),
T.ToTensor(),
T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])


device = ("cuda:0" if torch.cuda.is_available() else "cpu")
