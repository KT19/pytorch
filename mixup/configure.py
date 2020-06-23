#-*-coding:utf-8-*-
import torch
import torchvision
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device",default="cuda:0")
parser.add_argument("--lr",type=float,default=0.1)
parser.add_argument("--wd",type=float, default=5e-4)
parser.add_argument("--momentum",type=float,default=0.9)
parser.add_argument("--batch_size",type=int,default=100)
parser.add_argument("--n_epochs",type=int,default=100)
parser.add_argument("--alpha",type=float,default=0.2,help="hyper params in mixup")
parser.add_argument("--mode",default="mixup")

#analyze argument
args = parser.parse_args()

transform = transforms.Compose(
[
transforms.RandomCrop(32, 4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

preprocess = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
