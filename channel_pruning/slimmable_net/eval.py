import torch
import torchvision
from tqdm import tqdm
from config import device

def eval(model, testloader, switchable_list):
    model.eval()
    acc_dict = {}
    with torch.no_grad():
        for width in switchable_list:
            print("Evaluation on width = {}".format(width))
            total = 0
            acc = 0
            for img, target in tqdm(testloader):
                img = img.to(device)
                target = target.to(device)

                total += img.size(0)

                out = model(img, width)
                pred = torch.softmax(out, 1)
                pred = torch.argmax(pred, 1)
                acc += (pred == target).sum().item()

            acc_dict[width] = (acc / total)*100.

    return acc_dict
