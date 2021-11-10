import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(file: str, number_cities):
    file_path = create_path(file)
    headers = ["Iteration", "Time", "Mean", "Best"]
    for count in range(number_cities):
        headers.append("city" + str(count))
    df = pd.read_csv(file_path, header=None, names=headers)
    df = df.drop([0, 1])
    return df

def plot(file: str, num_cities):
    df = load_data(file, num_cities)

    plt.style.use('seaborn')
    fig, ax1 = plt.subplots()
    fig.suptitle("_")
    fig.set_size_inches(40 / 2.54, 30 / 2.54)

    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Mean", color='tab:red')
    ax1.plot(df["Iteration"], df["Mean"], color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(10))

    ax2 = ax1.twinx()

    ax2.set_ylabel("Best", color='tab:blue')
    ax2.plot(df["Iteration"], df["Best"], color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(10))

    plt.show()


def create_path(file: str):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, file)
    return file_path

if __name__ == "__main__":
    plot("r0123456.csv", 29)