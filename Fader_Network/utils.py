#-*-coding:utf-8-*-
import torch
import torchvision
import torchvision.transforms as transforms

IMG_SIZE = 64
ENC_LATENT = 256
ATTR = 40
mytransform = transforms.Compose([
transforms.CenterCrop(128),
transforms.Resize(64),
transforms.ToTensor(),
transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

device = ("cuda" if torch.cuda.is_available() else "cpu")
