#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from configure import *

def plot_results():
    save_file_path = "log_file"
    file_list = [save_file_path+"/iteration_"+str(i)+".csv" for i in range(1, args.iteration+1, 1)]

    plt.figure(figsize=(10, 6),dpi=180)

    for f in file_list:
        name = f.split("/")[1]
        name = name.split("_")[1]
        num = name.split(".")[0]

        df = pd.read_csv(f)
        if num == "1":
            label_name="original"
        else:
            label_name="pruned_"+num+"times"

        if label_name == "original":
            plt.plot(df["epoch"],df["accuracy"],marker="x",label=label_name)
        else:
            plt.plot(df["epoch"],df["accuracy"],marker="o",label=label_name)

    file = "random_init.csv"
    df = pd.read_csv(save_file_path+"/"+file)
    label_name = "randomly initialized"
    plt.plot(df["epoch"],df["accuracy"],linestyle="dashed",label=label_name)

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("results on mlp (dataset is mnist)")
    plt.grid()
    plt.legend()
    plt.savefig("results.png")

if __name__=="__main__":
    plot_results()
