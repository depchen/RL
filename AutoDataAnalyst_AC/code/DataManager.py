# coding: utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import RL as RD


# 数据管理类
class DataManager(object):

    # 该类用于管理数据，包括训练，测试。其中训练集分为训练数据和得分数据（用于Reward）。
    # 为此该类实现：
    # 1.储存数据（包括pickle）
    # 2.分割数据（分割train_data，使其变为训练数据和得分数据）

    def __init__(self):
        self.data_cv = None
        path = "/home/shawn/PycharmProjects/AutoDataAnalyst/MNIST_data/"
        # 训练集文件
        self.train_images_idx3_ubyte_file = path + 'train-images.idx3-ubyte'
        # 训练集标签文件
        self.train_labels_idx1_ubyte_file = path + 'train-labels.idx1-ubyte'

        # 测试集文件
        self.test_images_idx3_ubyte_file = path + 't10k-images.idx3-ubyte'
        # 测试集标签文件
        self.test_labels_idx1_ubyte_file = path + 't10k-labels.idx1-ubyte'

        # 获取完整版的mnist数据集
        #self.read_data()

        # 获取成年人收入数据集
        # self.read_data(3)  # RD.load_Adult

        self.read_data(6)

    def decode_idx3_ubyte(self, idx3_ubyte_file):
        """
        解析idx3文件的通用函数
        :param idx3_ubyte_file: idx3文件路径
        :return: 数据集
        """
        # 读取二进制数据
        file_x = open(idx3_ubyte_file, 'rb')
        bin_data = file_x.read()
        file_x.close()

        # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
        offset = 0
        fmt_header = '>iiii'
        magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

        # 解析数据集
        image_size = num_rows * num_cols
        offset += struct.calcsize(fmt_header)
        fmt_image = '>' + str(image_size) + 'B'
        images = np.empty((num_images, num_rows * num_cols))
        for i in range(num_images):
            images[i] = (np.array(struct.unpack_from(fmt_image, bin_data, offset)) > 0).astype(int)
            offset += struct.calcsize(fmt_image)
        return images

    def decode_idx1_ubyte(self, idx1_ubyte_file):
        """
        解析idx1文件的通用函数
        :param idx1_ubyte_file: idx1文件路径
        :return: 数据集
        """
        # 读取二进制数据
        file_x = open(idx1_ubyte_file, 'rb')
        bin_data = file_x.read()
        file_x.close()

        # 解析文件头信息，依次为魔数和标签数
        offset = 0
        fmt_header = '>ii'
        magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
        # print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

        # 解析数据集
        offset += struct.calcsize(fmt_header)
        fmt_image = '>B'
        # labels = np.empty((num_images, 10))
        labels = np.empty((num_images,))
        for i in range(num_images):
            index = struct.unpack_from(fmt_image, bin_data, offset)[0]
            # labels[i] = np.zeros([10])
            # labels[i][int(index)] = 1
            labels[i] = int(index)
            offset += struct.calcsize(fmt_image)
        return labels

    def load_train_data(self):
        """
        :param idx_ubyte_file: idx文件路径
        :return: n*1维np.array对象，n为图片数量
        """
        train_data, train_labels = self.decode_idx3_ubyte(self.train_images_idx3_ubyte_file), self.decode_idx1_ubyte(self.train_labels_idx1_ubyte_file)

        return train_data, train_labels

    def load_test_data(self):
        """
        :param idx_ubyte_file: idx文件路径
        :return: n*row*col维np.array对象，n为图片数量
        """
        test_data, test_labels = self.decode_idx3_ubyte(self.test_images_idx3_ubyte_file), self.decode_idx1_ubyte(self.test_labels_idx1_ubyte_file)

        return test_data, test_labels

    # 从指定文件读取数据
    def read_data(self, dataset_id=None):
        if dataset_id == None:
            if dataset_id == None:
                train_data, train_labels = self.load_train_data()
                test_data, test_labels = self.load_test_data()
                pi = np.random.permutation(len(train_data))
                train_data, train_labels = train_data[pi], train_labels[pi]
                pi = np.random.permutation(len(test_data))
                test_data, test_labels = test_data[pi], test_labels[pi]
                self.data_cv = {'data_cv': train_data,
                                'labels_cv': train_labels,
                                'data_test': test_data,
                                'labels_test': test_labels
                                }
                print("data_cv:", np.shape(self.data_cv['data_cv']), 'labels_cv:', np.shape(self.data_cv['labels_cv']))
                print("data_test:", np.shape(self.data_cv['data_test']), 'labels_test:', np.shape(self.data_cv['labels_test']))

        else:
            #                   0,                  1,                      2,                  3,              4,              5,                          6
            dataset = [skdata.load_iris, skdata.load_digits, skdata.load_breast_cancer, RD.load_Adult, RD.load_Mnist, RD.load_Car_Evaluation, RD.load_litter_recognition_data_set]
            data, labels = dataset[dataset_id](return_X_y=True)
            for _ in range(20):
                pi = np.random.permutation(len(data))
                data, labels = data[pi], labels[pi]

            self.data_cv = {'data_cv': data[:int(len(data) * 0.7)],
                            'labels_cv': labels[:int(len(data) * 0.7)],
                            'data_test': data[int(len(data) * 0.7):],
                            'labels_test': labels[int(len(data) * 0.7):]
                            }
            print("data_cv:", np.shape(self.data_cv['data_cv']), 'labels_cv:', np.shape(self.data_cv['labels_cv']))
            print("data_test:", np.shape(self.data_cv['data_test']), 'labels_test:', np.shape(self.data_cv['labels_test']))


def test():
    datamanager = DataManager()
    data = datamanager.data
    print("Train:", np.shape(data['train_data']), np.shape(data['train_labels']))
    print("Test:", np.shape(data['test_data']), np.shape(data['test_labels']))

    im = np.array(data['train_data'][20])
    im = im.reshape(28, 28)
    print(im)
    print(data['train_labels'][20])
    plt.imshow(im, cmap='gray')
    plt.show()

    im = np.array(data['test_data'][30])
    im = im.reshape(28, 28)
    print(im)
    print(data['test_labels'][30])
    plt.imshow(im, cmap='gray')
    plt.show()

    print('done')

if __name__ == "__main__":
    test()
