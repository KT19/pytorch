#-*-coding:utf-8-*-
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

def plot():
    df = pd.read_csv("log_state.csv")
    swithcable_list = [0.25, 0.5, 0.75, 1.0]

    fig = plt.figure()

    for l in swithcable_list:
        plt.plot(df["epoch"],df[str(l)],label="size_"+str(l))
        print("final accuracy in x{} = {}[%]".format(l, max(df[str(l)])))

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend()
    fig.savefig("plot_result.png")


if __name__ == "__main__":
    plot()
