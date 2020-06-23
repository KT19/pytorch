#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # -----(1)
import matplotlib.pyplot as plt

def show_graph():
    df = pd.read_csv("mixup_train_log.csv")
    plt.plot(df["EPOCH"],df["ACCURACY"],label="mixup")
    print("max accuracy with mixup: {}".format(max(df["ACCURACY"])))

    df = pd.read_csv("normal_train_log.csv")
    plt.plot(df["EPOCH"],df["ACCURACY"],label="no_mixup")
    print("max accuracy without mixup: {}".format(max(df["ACCURACY"])))

    plt.grid()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("results of mixup")
    plt.legend()
    plt.savefig("results.png")

if __name__ =="__main__":
    show_graph()
