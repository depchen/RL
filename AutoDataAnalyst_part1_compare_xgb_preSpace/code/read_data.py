# coding: utf-8
# 读取各个数据集并做简单预处理
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import os


def class_map_encode(feature):
    class_mapping = {label: idx for idx, label in enumerate(set(feature))}
    return feature.map(class_mapping)


def one_hot(feature):
    encode = pd.get_dummies(feature, prefix=None, prefix_sep='_', dummy_na=False,
                            columns=None, sparse=False, drop_first=False)
    return encode


def see_null(df):
    return df.isnull().any()


def see_feature_null(df, feature):
    return df[df[feature].isnull()]


def load_Adult(return_X_y=True, root="datasets"+os.sep+"Adult"):
    # 1 3 5 6 7 8 9 13 14
    train_file_name = root + os.sep + "adult.data"
    test_file_name = root + os.sep + "adult.test"

    train_data = pd.read_csv(train_file_name, header=None)
    test_data = pd.read_csv(test_file_name, header=None)

    train_data[train_data[1] == " ?"] = np.nan
    train_data = train_data.dropna()
    length = train_data.count()[0]
    test_data[test_data[1] == " ?"] = np.nan
    test_data = test_data.dropna()
    all_data = pd.concat([train_data, test_data])
    for f in [1,3,5,6,7,8,9,13]:
        encode = one_hot(all_data.loc[:, f])
        del all_data[f]
        all_data = pd.concat([all_data, encode], axis=1)
    all_data.loc[(all_data[14] == " <=50K") | (all_data[14] == " <=50K."), 14] = 0
    all_data.loc[(all_data[14] == " >50K") | (all_data[14] == " >50K."), 14] = 1
    all_data["labels"] = all_data[14]
    del all_data[14]

    train_data = all_data.iloc[:length, :]
    test_data = all_data.iloc[length+1:, :]

    if return_X_y:
        y = np.array(all_data["labels"].values, np.int)
        del all_data["labels"]
        X = np.array(all_data.values, np.float32)
        return X, y
    else:
        return train_data, test_data


def load_Car_Evaluation(return_X_y=True, root="datasets"+os.sep+"Car_Evaluation"):
    train_file_name = root + os.sep + "car.data"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data.loc[train_data[6] == "unacc", 6] = 0
    train_data.loc[train_data[6] == "acc", 6] = 1
    train_data.loc[train_data[6] == "good", 6] = 2
    train_data.loc[train_data[6] == "vgood", 6] = 3
    labels = train_data[6]
    del train_data[6]
    train_data = one_hot(train_data)

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data


def load_litter_recognition_data_set(return_X_y=True, root="datasets"+os.sep+"litter_recognition_data_set"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    train_data.loc[train_data[0] == "A", 0] = 0
    train_data.loc[train_data[0] == "B", 0] = 1
    train_data.loc[train_data[0] == "C", 0] = 2
    train_data.loc[train_data[0] == "D", 0] = 3
    train_data.loc[train_data[0] == "E", 0] = 4
    train_data.loc[train_data[0] == "F", 0] = 5
    train_data.loc[train_data[0] == "G", 0] = 6
    train_data.loc[train_data[0] == "H", 0] = 7
    train_data.loc[train_data[0] == "I", 0] = 8
    train_data.loc[train_data[0] == "J", 0] = 9
    train_data.loc[train_data[0] == "K", 0] = 10
    train_data.loc[train_data[0] == "L", 0] = 11
    train_data.loc[train_data[0] == "M", 0] = 12
    train_data.loc[train_data[0] == "N", 0] = 13
    train_data.loc[train_data[0] == "O", 0] = 14
    train_data.loc[train_data[0] == "P", 0] = 15
    train_data.loc[train_data[0] == "Q", 0] = 16
    train_data.loc[train_data[0] == "R", 0] = 17
    train_data.loc[train_data[0] == "S", 0] = 18
    train_data.loc[train_data[0] == "T", 0] = 19
    train_data.loc[train_data[0] == "U", 0] = 20
    train_data.loc[train_data[0] == "V", 0] = 21
    train_data.loc[train_data[0] == "W", 0] = 22
    train_data.loc[train_data[0] == "X", 0] = 23
    train_data.loc[train_data[0] == "Y", 0] = 24
    train_data.loc[train_data[0] == "Z", 0] = 25

    labels = train_data[0]
    del train_data[0]

    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data

def load_epileptic_recognition(return_X_y=True, root="datasets"+os.sep+"epileptic_Recognition_Data_Set"):
    train_file_name = root + os.sep + "data.csv"
    train_data = pd.read_csv(train_file_name, header=None)
    labels=train_data[0]
    del train_data[0]
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data

def load_Mushroom(return_X_y=True, root="datasets"+os.sep+"Mushroom"):
    train_file_name = root + os.sep + "agaricus-lepiota.data"
    train_data = pd.read_csv(train_file_name, header=None)

    train_data.loc[train_data[0]=="e", 0] = 0
    train_data.loc[train_data[0]=="p", 0] = 1
    labels = train_data[0]
    del train_data[0]
    train_data = one_hot(train_data)
    if return_X_y:
        return np.array(train_data.values, np.float32), np.array(labels.values, np.int)
    else:
        train_data["labels"] = labels
        return train_data

def load_mnist(return_X_y=True):
    mnist_data = input_data.read_data_sets('./datasets/mnist1', one_hot=True)
    train_data = mnist_data.train.images[:7000, :]
    train_data_label = mnist_data.train.labels[:7000, :]
    test_data = mnist_data.test.images[:3000, :]
    test_data_lable = mnist_data.test.labels[:3000, :]
    all_data = np.concatenate([train_data, test_data])
    all_data_lable=np.concatenate([train_data_label,test_data_lable])
    all_data_lable = np.array(all_data_lable).reshape(10000, 10)
    all_lable = []
    for i in range(len(all_data_lable)):
        all_lable.append(list(all_data_lable[i]).index(1))
    if return_X_y:
        y = np.array(all_lable, np.int)
        x=np.array(all_data,np.float32)
        return x,y
    else:
        return all_data, all_lable

def load_Mnist(return_X_y=True, root = "datasets" + os.sep + "mnist"):
    train_file_name = root + os.sep + "optdigits.tra"
    test_file_name = root + os.sep + "optdigits.tes"

    train_data = pd.read_csv(train_file_name, header=None)
    test_data = pd.read_csv(test_file_name, header=None)

    all_data = pd.concat([train_data, test_data])

    all_labels = all_data.loc[:, 64]
    all_data = all_data.loc[:, :63]

    if return_X_y:
        y = np.array(all_labels.values, np.int)
        del all_labels
        X = np.array(all_data, np.float32)
        del all_data
        return X, y
    else:
        return train_data, test_data


if __name__ == "__main__":
    # root = "E:\\PySpace\\datasets\\Mushroom"
    X, y = load_litter_recognition_data_set(return_X_y=True)
    print(X.shape)
    print(y.shape)
    pi = np.random.permutation(len(X))
    X, y = X[pi], y[pi]
    print(X[0:2])
    print(y[0:2])
