#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import random
import os
import torchvision
from PIL import Image
from tqdm import tqdm
from configure import *
from modules import *
from data_loader import *
#for time
import time

def train():
    def get_resized_input(x,size):
        y = F.interpolate(x,size=(size,size),mode="bilinear",align_corners=True)

        return y

    def reset_grad():
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

    def gradient_penalty(real_data, x_hat):
        """
        real_data:
        x_hat:
        """
        eps = torch.rand(x_hat.size(0),device=device).view(-1,1,1,1)
        x_gp = eps*real_data+(1.0 - eps)*x_hat
        x_gp = autograd.Variable(x_gp, requires_grad=True)

        disc_interpolate = discriminator(x_gp)
        gradient = autograd.grad(
        outputs=disc_interpolate,inputs=x_gp,
        grad_outputs = torch.ones(disc_interpolate.size(),device=device),
        create_graph=True,retain_graph=True,only_inputs=True)[0]

        gradient = gradient.view(gradient.size(0),-1) #flatten

        grad_penalty = ((gradient.norm(2, dim=1) - 1)**2).mean()

        return grad_penalty

    """load model"""
    generator = Generator(model_num=args.model_num,inc=args.inc,scale_up_step=args.sus).train().to(device)
    discriminator = Discriminator(inc=args.inc,scale_up_step=args.sus).train().to(device)

    """set optimizer"""
    g_optimizer = optim.Adam(generator.models[generator.current_scale_num-1].parameters(),lr=args.g_lr,betas=(0.5,0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),lr=args.d_lr,betas=(0.5,0.999))

    #get customdata
    trainloader = get_loader(image_dir=args.data_dir+args.img_name, batch_size=args.batch, preprocess=preprocess, to_gray=to_gray, to_rgb=to_rgb, mode="train", ref_img=args.ref_img)

    #criterion
    recon_loss = nn.MSELoss()

    model_num = args.model_num
    n_iter = args.n_iter
    log_per_step = (n_iter // 100)
    prog_itr = n_iter//model_num
    decay_itr = int(prog_itr*0.8)
    prev_itr = 0
    noise_size = generator.size_list[0]
    print("log per iteration is {}".format(log_per_step))
    print("progressive is per {}".format(prog_itr))
    print("model num is {}".format(model_num))
    print("start train")

    #set fixed noise
    noise = torch.randn(args.batch, 1, noise_size, noise_size, device=device)
    fixed_noise = [noise.repeat(1,3,1,1)]
    zero_list = [torch.zeros(args.batch,3,generator.size_list[i],generator.size_list[i],device=device) for i in range(1,model_num)]
    z_list = fixed_noise + zero_list

    total_d_loss = []
    total_g_loss = []

    color_imgs,_ = next(iter(trainloader))
    color_imgs = color_imgs.to(device)

    generator.set_img_list(color_imgs) #for calculating rmse

    print("--start with time measurement---")
    start = time.time()
    for itr in range(1,n_iter+1,1):
        if itr % log_per_step == 0:
            print("Iter:[{}]/[{}]".format(itr,n_iter))

        prev_itr += 1
        if prev_itr == decay_itr:
            for param in g_optimizer.param_groups:
                param["lr"] *= 0.1
            print("lr in generator is updated")

            for param in d_optimizer.param_groups:
                param["lr"] *= 0.1
            print("lr in discriminator is updated")


        """train discriminator"""
        temp = []
        for i in range(args.d_step):
            reset_grad()
            """train fake"""
            reconstruct_outputs = generator(z_list)
            #calculate rmse
            rmses = [1.0]
            for j in range(generator.current_scale_num-1):
                rmses.append(F.mse_loss(reconstruct_outputs[j],get_resized_input(color_imgs, reconstruct_outputs[j].size(2))).item())

            #scaled noise
            noise_list = []
            for j in range(len(rmses)):
                noise = torch.randn(args.batch,1,generator.size_list[j],generator.size_list[j],device=device)
                noise = noise.repeat(1,3,1,1)
                noise_list.append(rmses[j]*noise)

            outputs = generator(noise_list)

            d_fake = discriminator(outputs[-1])

            """train real"""
            real_data = get_resized_input(color_imgs,outputs[-1].size(2))
            d_real = -discriminator(real_data)

            """calc gp"""
            gp = gradient_penalty(real_data,outputs[-1])

            d_loss = torch.mean(d_fake) + torch.mean(d_real) + args.gp_l*gp
            d_loss.backward()
            d_optimizer.step()
            temp.append(d_loss.item())

        total_d_loss.append(np.mean(temp))

        """train generator"""
        temp = []
        for i in range(args.g_step):
            reset_grad()
            reconstruct_outputs = generator(z_list)
            #calculate rmse
            rmses = [1.0]
            for j in range(generator.current_scale_num-1):
                rmses.append(F.mse_loss(reconstruct_outputs[j],get_resized_input(color_imgs, reconstruct_outputs[j].size(2))).item())

            #scaled noise
            noise_list = []
            for j in range(len(rmses)):
                noise = torch.randn(args.batch,1,generator.size_list[j],generator.size_list[j],device=device)
                noise = noise.repeat(1,3,1,1)
                noise_list.append(rmses[j]*noise)

            outputs = generator(noise_list)
            outputs = outputs[-1] #get last scale
            g_real = discriminator(outputs)

            origin = reconstruct_outputs[-1]
            #wgan loss + rmses
            g_loss = -torch.mean(g_real) + args.lamb*F.mse_loss(reconstruct_outputs[-1],get_resized_input(color_imgs,reconstruct_outputs[-1].size(2)))
            g_loss.backward()
            g_optimizer.step()
            temp.append(g_loss.item())

        total_g_loss.append(np.mean(temp))

        if itr % log_per_step == 0:
            total_d_loss = np.sum(total_d_loss)
            total_g_loss = np.sum(total_g_loss)
            print("d loss:[{}], g loss:[{}]".format(total_d_loss,total_g_loss))
            total_d_loss = []
            total_g_loss = []

            save_path = "one_shot/"
            os.makedirs(save_path+args.img_name,exist_ok=True)

            torchvision.utils.save_image(0.5*real_data+0.5,save_path+args.img_name+"/ground_truth.png")
            torchvision.utils.save_image(0.5*outputs+0.5,save_path+args.img_name+"/output"+str(itr)+".png")
            torchvision.utils.save_image(0.5*origin+0.5,save_path+args.img_name+"/origin.png")
            if generator.current_scale_num > 1:
                torchvision.utils.save_image(0.5*reconstruct_outputs[-2]+0.5,save_path+args.img_name+"/prev.png")

            os.makedirs("save_model/"+args.img_name,exist_ok=True)
            generator.to("cpu")
            torch.save(generator.state_dict(),"save_model/"+args.img_name+"/generator.pth")
            generator.to(device)

            discriminator.to("cpu")
            torch.save(discriminator.state_dict(),"save_model/"+args.img_name+"/discriminator.pth")
            discriminator.to(device)

        if itr % prog_itr == 0 and itr < n_iter:
            print("progressive")
            generator.scale_up()
            discriminator.scale_up()

            #set new optimizer
            g_optimizer = optim.Adam(generator.models[generator.current_scale_num-1].parameters(),lr=args.g_lr,betas=(0.5,0.999))
            d_optimizer = optim.Adam(discriminator.parameters(),lr=args.d_lr,betas=(0.5,0.999))
            prev_itr = 0

    elapsed_time = time.time() - start
    print("elapsed_time:{}".format(elapsed_time/60.) + "[min]")


if __name__ == "__main__":
    train()
