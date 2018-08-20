import numpy as np
import pandas as pd
import os

#data = np.loadtxt('/home/shawn/PycharmProjects/AutoDataAnalyst/AutoDataAnalyst_v1/code/datasets/mnist/optdigits.tes.txt')
add = '/home/shawn/PycharmProjects/AutoDataAnalyst/AutoDataAnalyst_v1/code/datasets/mnist/optdigits.tes.txt'

root = "datasets"+os.sep+"mnist"
train_file_name = root + os.sep + "optdigits.tra"
test_file_name = root + os.sep + "optdigits.tes"

train_data = pd.read_csv(train_file_name, header=None)
test_data = pd.read_csv(test_file_name, header=None)

print("train_data.shape:", train_data.shape)
print("test_data.shape:", test_data.shape)
