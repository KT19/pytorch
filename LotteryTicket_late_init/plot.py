#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configure import *

def plot_results():
    save_file_path = "log_file"
    ratio = [10, 20, 30, 40, 50, 60, 70, 80, 90, 92, 95, 98, 99]

    f = save_file_path+"/unpruned_model.csv"
    df = pd.read_csv(f)
    base = [max(df["accuracy"]) for _ in ratio]
    winning = []
    random = []
    params = []
    for r in ratio:
        f = save_file_path+"/pruned_"+str(r)+"_percent.csv"
        f2 = save_file_path+"/pruned_"+str(r)+"_percent_with_random_init.csv"

        df = pd.read_csv(f) #read csv file
        df2 = pd.read_csv(f2) #read csv file

        winning.append(max(df["accuracy"]))
        random.append(max(df2["accuracy"]))
        params.append(r)

    plt.plot(params, winning, marker="o",label="winning ticket")
    plt.plot(params, random, marker="x",label="random init")
    plt.plot(params, base, label="base(dense model)", linestyle="dashed")

    plt.xlabel("pruned_params")
    plt.ylabel("accuracy")
    plt.xticks([i for i in range(10, 101,10)])
    plt.title("results of lottery ticket(late init)")
    plt.grid()
    plt.legend()
    plt.savefig("results.png")

if __name__=="__main__":
    plot_results()
