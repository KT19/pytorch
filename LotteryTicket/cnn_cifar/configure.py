#-*-coding:utf-8-*-
import torch
import torchvision
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--iteration", type=int, default=5)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--ratio", type=float,default=0.95)
parser.add_argument("--lr",type=float,default=0.1)
parser.add_argument("--device",default="cuda:0")

args = parser.parse_args()

device = (args.device if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(
mean=(0.5,0.5,0.5),
std=(0.5,0.5,0.5)
)])
