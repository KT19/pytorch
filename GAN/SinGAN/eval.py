#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torchvision
from PIL import Image
from tqdm import tqdm
from configure import *
from modules import *
from data_loader import *

def resized_input(x,s1,s2):
    y = F.interpolate(x,size=(s1,s2),mode="bilinear",align_corners=True)

    return y

def eval():
    save_path = "eval_img"
    os.makedirs("eval_img",exist_ok=True)

    #load model
    generator = Generator(model_num=args.model_num,inc=args.inc,scale_up_step=args.sus,scale=args.scale).eval()
    generator.load_state_dict(torch.load("save_model/"+args.model+"/generator.pth"))
    generator.to(device)
    generator.current_scale_num = args.model_num

    #dataloader
    testloader = get_loader(image_dir=args.data_dir+args.img_name, batch_size=args.batch, preprocess=preprocess_test, to_gray=to_gray, to_rgb=to_rgb, mode="test", ref_img=args.ref_img)
    color_img, _ = next(iter(testloader))
    color_img = color_img.to(device)
    scale = args.scale
    eval_noise_scale = 100.

    while True:
        with torch.no_grad():
            noise_size = [16,20]
            """get rmse"""
            fixed_z = torch.randn(1,1,noise_size[0],noise_size[1],device=device)
            fixed_z = fixed_z.repeat(1,3,1,1)
            z_list = [fixed_z]

            for i in range(generator.current_scale_num-1):
                noise_size[0] = int(scale*noise_size[0])
                noise_size[1] = int(scale*noise_size[1])
                z_list.append(torch.zeros(1,3,noise_size[0],noise_size[1],device=device))

            reconstruct_outputs = generator(z_list,eval=True)

            #calculate rmse
            rmses = [1.0]
            for j in range(generator.current_scale_num-1):
                rmses.append(F.mse_loss(reconstruct_outputs[j],resized_input(color_img, reconstruct_outputs[j].size(2),reconstruct_outputs[j].size(3))).item()/eval_noise_scale)

            #scaled noise
            noise_list = []
            for j in range(len(rmses)):
                noise = torch.randn(args.batch,1,z_list[j].size(2),z_list[j].size(3),device=device)
                noise = noise.repeat(1,3,1,1)
                noise_list.append(rmses[j]*noise)

            outputs = generator(noise_list,eval=True)

            torchvision.utils.save_image(0.5*color_img+0.5,save_path+"/ground_truth.png")
            for s,o in enumerate(outputs):
                torchvision.utils.save_image(0.5*o+0.5,save_path+"/output_scale"+str(s+1)+".png")

            _ = input("enter next")

if __name__ == "__main__":
    eval()
