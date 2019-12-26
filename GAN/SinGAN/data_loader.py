#-*-config:utf-8-*-
import torch
import torch.nn as nn
from torch.utils import data
from PIL import Image
import os

class MyDataset(data.Dataset):
    def __init__(self, image_dir, preprocess, to_gray, to_rgb, mode="train",ref_img="img1.jpg"):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.to_gray = to_gray
        self.to_rgb = to_rgb
        self.mode = mode
        self.ref_img = ref_img

        self.dataset = []

        self.get_images() #get file

    def get_images(self):
        for t in range(100):
                self.dataset.append(self.ref_img)

    def __getitem__(self,index):
        file_name = self.dataset[index]

        image = Image.open(os.path.join(self.image_dir,file_name))
        image = self.preprocess(image)

        return self.to_rgb(image), self.to_gray(image)

    def __len__(self):
        return len(self.dataset)

def get_loader(image_dir, batch_size, preprocess, to_gray, to_rgb, mode="train",ref_img="img1.jpg",num_workers=1):
    dataset = MyDataset(image_dir, preprocess=preprocess, to_gray=to_gray, to_rgb=to_rgb, mode=mode, ref_img=ref_img)
    data_loader = data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    return data_loader
