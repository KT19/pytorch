#-*-coding:utf-8-*-
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import warnings
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--g_lr",type=float,default=0.0005)
parser.add_argument("--d_lr",type=float,default=0.0005)
parser.add_argument("--device",default="cuda:1")
parser.add_argument("--lamb",type=float,default=10.)
parser.add_argument("--batch",type=int,default=1)
parser.add_argument("--img_name",default="path to img dir")
parser.add_argument("--ref_img",default="test.jpg")
parser.add_argument("--model",default="test")
parser.add_argument("--data_dir",default="path to dataset dir")
parser.add_argument("--n_iter",type=int,default=20000)
parser.add_argument("--model_num",type=int,default=10)
parser.add_argument("--inc",type=int,default=32)
parser.add_argument("--sus",type=int,default=4)
parser.add_argument("--scale",type=float,default=1.3)
parser.add_argument("--gp_l",type=float,default=0.1)
parser.add_argument("--d_step",type=int,default=3)
parser.add_argument("--g_step",type=int,default=3)

args = parser.parse_args()
device = (args.device if torch.cuda.is_available() else "cpu")

IMG_SIZE = 256

preprocess = transforms.Compose(
[
transforms.Resize((IMG_SIZE,IMG_SIZE)),
]
)

preprocess_test = transforms.Compose([
transforms.Resize((IMG_SIZE,IMG_SIZE)),
])
to_gray = transforms.Compose([
transforms.Grayscale(3),
transforms.ToTensor(),
transforms.Normalize(
mean = (0.5, 0.5, 0.5),
std = (0.5, 0.5, 0.5)
)])

to_rgb = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(
mean= (0.5,0.5,0.5),
std = (0.5,0.5,0.5)
)])
