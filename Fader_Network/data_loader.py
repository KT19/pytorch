#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import random
import os
from utils import *

class CelebA(data.Dataset):
    def __init__(self, image_dir, attr_path, transform, mode):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == "train":
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path,"r")]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for v in values:
                if(v == "1"):
                    label.append(1)
                else:
                    label.append(0)

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        print("Finish Preprocessing")


    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == "train" else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), np.asarray(label,dtype=np.float32)

    def __len__(self):
        return self.num_images

def get_loader(image_dir, attr_path, batch_size=64, mode="train", num_workers=1):
    dataset = CelebA(image_dir, attr_path, mytransform, mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader,dataset.attr2idx
