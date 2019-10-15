#-*-coding:utf-8-*-
from config import *
from train import *
from eval import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode",default="train")

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        eval()
    else:
        raise "Not implemented error"
