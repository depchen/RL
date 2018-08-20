from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np


mnist_data = input_data.read_data_sets('./datasets/mnist1', one_hot=True)
train_data = mnist_data.train.images[:7000, :]
train_data_label = mnist_data.train.labels[:7000, :]
test_data = mnist_data.test.images[:3000, :]
test_data_lable = mnist_data.test.labels[:3000, :]
all_data = np.concatenate([train_data, test_data])
all_data_lable = np.concatenate([train_data_label, test_data_lable])
all_data_lable=np.array(all_data_lable).reshape(10000,10)
all_label=np.where(all_data_lable==np.max(all_data_lable))
# all_lable=[]
# for i in range(len(all_data_lable)):
#     all_lable.append(list(all_data_lable[i]).index(1))
if return_X_y:
    y = np.array(train_data, np.int)
    del all_labels
    X = np.array(all_data, np.float32)
    del all_data
