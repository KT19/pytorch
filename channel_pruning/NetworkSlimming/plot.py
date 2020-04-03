#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot():
    df = pd.read_csv("train_log.csv")

    plt.plot(df["epoch"],df["accuracy"])
    plt.savefig("results.png")

if __name__ =="__main__":
    plot()
