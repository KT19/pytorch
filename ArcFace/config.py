#-*-coding:utf-8-*-
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, ), (0.5,))])

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
