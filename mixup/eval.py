#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from tqdm import tqdm
from configure import *

class Eval():
    def __init__(self):
        print("set data in evaluation")
        testset = torchvision.datasets.CIFAR10(root="~/dataset",train=False,transform=preprocess,download=True)

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

    def eval(self, model):
        print("evaluation")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, labels in tqdm(self.testloader):
                total += img.size(0)
                img = img.to(device)
                outputs = model(img)

                outputs = torch.softmax(outputs,1) #category
                pred_cls = torch.argmax(outputs,1)

                for i in range(outputs.size(0)):
                    if pred_cls[i].item() == labels[i].item():
                        correct += 1
        model.train()

        return (correct / total) * 100.
