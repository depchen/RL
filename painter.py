import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import argparse

class Painter():
    """docstring for Paiter."""
    def __init__(self, category='line',
                dir = None,
                colorful = 'favorite',
                axis = 0):
        self.colorful = colorful
        self.category = category
        self.color_mark = 0
        self.data = {}
        self.dir = dir
        self.axis = axis

        sns.set(style="whitegrid", color_codes=True)
        if dir == '':
            self.data['random-1'] = self.random_data()
            self.data['random-2'] = self.random_data() - 2
            self.data['random-3'] = self.random_data() - 4
        else:
            files = os.listdir(dir)
            self.wanted = [i for i in files if i[-3:] == 'csv']
            self.read_data()

        if self.colorful == 'favorite':
            colors = ["blue","orange","violet","green","red"]
            shadow_colors = ["light blue","light orange","light violet","light green","baby pink"]
            self.color_palette = sns.xkcd_palette(colors)
            self.shadow_color = sns.xkcd_palette(shadow_colors)

    def read_data(self):
        if len(self.wanted) == 0:
            print('No valied files {only .csv supported}')
            return
        for name in self.wanted:
            print(self.dir + name)
            d = pd.read_csv(self.dir + name, encoding='utf-8')
            if self.axis == 1: d = d.T
            self.data[name[:-4]] = d
            print("All %d files will be ploted." % len(self.wanted))

    def random_data(self, times = 5):
        return pd.DataFrame([self.random_feature() + np.random.randint(5) for i in range(times)])

    def random_feature(self, length = 500):
        x = np.arange(length)
        y = np.log(x + 1) + np.random.standard_normal(length)
        return y

    def colorful_world(self, pair=True):
        k = self.color_mark
        self.color_mark += 1
        if self.color_mark >= len(self.color_palette):self.color_mark -= len(self.color_palette)
        if pair:
            return self.color_palette[k], self.shadow_color[k]
        else:
            return self.color_palette[k]

    def painte(self, gamma=0.6, stretch=10, shadow=True):
        with self.color_palette:
            if self.category == 'line':
                for name, each_line in self.data.items():
                    self._plot_line(gamma, each_line, name, stretch, shadow)
                plt.xlabel('Frames')
                plt.ylabel('Rewards')
                plt.legend(loc='upper left')
                plt.savefig(self.dir + name + ".jpg")
                plt.show()


    def _plot_line(self, gamma, data, name, stretch,shadow=True):
        mean = np.mean(data, axis=0)
        c, sc = self.colorful_world()
        for i in range(1,len(mean)):
            mean[i] = mean[i] * (1 - gamma) + mean[i-1] *  gamma
        plt.plot(stretch*np.arange(len(mean)), mean, color=c, label=name)
        if shadow:
            std = np.std(data, axis=0)
            for i in range(1,len(std)):
                std[i] = std[i] * (1 - gamma) + std[i-1] * gamma
            plt.fill_between(stretch*np.arange(len(mean)), mean-std, mean+std, facecolor=sc, alpha=0.4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-k', '--kind', default='line', type=str, help='choose the kind of plot: { line/scatter }')
    parser.add_argument('-sh', '--shadow', default=True, type=bool, help='mean +- std | error area: { True/False }')
    parser.add_argument('-st', '--stretch', default=10, type=int, help='times of x axis stretch: int')
    parser.add_argument('-d', '--data_dir', default='', type=str, help='CSV data files directory: .../target/')
    parser.add_argument('-g', '--gamma', default=0.6, type=float, help='discount of smooth')
    args = parser.parse_args()

    kind = args.kind
    shadow = args.shadow
    dir = args.data_dir
    stretch = args.stretch
    gamma = args.gamma

    p = Painter(dir = dir, category = kind)
    p.painte(stretch = stretch, shadow = shadow, gamma = gamma)
