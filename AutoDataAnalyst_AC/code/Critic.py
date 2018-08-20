# coding:utf-8
import tensorflow as tf
import numpy as np
import time
from collections import deque

from RL.AutoDataAnalyst_AC.code.configFile.AgentConfigFile import AgentConfig as Config


class Critic:
    def __init__(self, params):
        GAMMA = 0.9
        lr=0.01
        self.step=len(params)
        self.s = tf.placeholder(tf.float32, [Config.batch_size, self.step], "state")
        self.v_ = tf.placeholder(tf.float32, [Config.batch_size, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, sess,s, r, s_):
        s=np.array(s).reshape(Config.batch_size,self.step)
        s_ = np.array(s_).reshape(Config.batch_size, self.step)
        r=np.mean(r)
        v_ = sess.run(self.v, {self.s: s_})
        td_error, _ = sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error






    # def __init__(self, params):
    #     # tf.reset_default_graph()
    #     #t1=tf.Graph()
    #     self.n_step = len(params)  # 该算法需要优化的超参数的个数
    #     self.GAMMA=0.9
    #     self.lr=0.01
    #
    #     self.y_=tf.placeholder(tf.float32,[Config.batch_size,1],name='y_')
    #     self.r = tf.placeholder(tf.float32, [Config.batch_size, 1], name='r')
    #
    #     # 1.输入层
    #     with tf.name_scope('input_layer1'):
    #         self.x = []  # 输入数据
    #         self.labels = []  # 输出数据对应的标签
    #         self.reward = tf.placeholder(tf.float32, [Config.batch_size, 1], name='reward1')  # 奖励值
    #
    #         for i in range(self.n_step):
    #             if i == 0:
    #                 x_temp = tf.placeholder(tf.float32, [Config.batch_size, params[-1]], name='input1_' + str(i + 1))
    #                 self.x.append(x_temp)
    #             else:
    #                 x_temp = tf.placeholder(tf.float32, [Config.batch_size, params[i - 1]], name='input1_' + str(i + 1))
    #                 self.x.append(x_temp)
    #             labels_temp = tf.placeholder(tf.int32, [Config.batch_size, 1], name='labels1_' + str(i + 1))
    #             self.labels.append(labels_temp)
    #
    #     # 2.输入全连接层；因为LSTM每个时刻输入数据的维度不一致，添加此层的主要目的是使输入到LSTM_cell层的数据维度统一
    #     with tf.name_scope('layer_11'):
    #         # X_in = X*W + b 输入输出断的参数，要不要设置为一样的值？？？
    #         weights_in = []  # 输入层与LSTM_cell层之间的连接权重
    #         biases_in = []  # LSTM_cell层神经元的偏置参数
    #         for i in range(self.n_step):
    #             if i == 0:
    #                 weights_in_temp = tf.Variable(tf.random_uniform([params[-1], Config.n_hidden_units], -0.2, 0.2),
    #                                               trainable=True, name="weights_in1_" + str(i + 1))
    #                 weights_in.append(weights_in_temp)
    #             else:
    #                 weights_in_temp = tf.Variable(tf.random_uniform([params[i - 1], Config.n_hidden_units], -0.2, 0.2),
    #                                               trainable=True, name="weights_in1_" + str(i + 1))
    #                 weights_in.append(weights_in_temp)
    #             biases_in_temp = tf.Variable(tf.constant(0.1, shape=[Config.n_hidden_units]), trainable=True,
    #                                          name="biases_in1_" + str(i + 1))
    #             biases_in.append(biases_in_temp)
    #         input_lstm = []  # LSTM_cell层的输入
    #         for i in range(self.n_step):
    #             input_lstm_temp = tf.nn.bias_add(tf.matmul(self.x[i], weights_in[i]), biases_in[i])
    #             input_lstm.append(input_lstm_temp)
    #
    #     # 3.循环神经网络层（核心层）
    #     with tf.variable_scope('LSTM_cell1'):
    #         # 3层 每层35个单元
    #         stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    #             [tf.contrib.rnn.BasicLSTMCell(Config.n_hidden_units, forget_bias=1.0, state_is_tuple=True) for _ in
    #              range(Config.n_layers)])
    #         state = stacked_lstm.zero_state(Config.batch_size, tf.float32)
    #         output = []  # LSTM_cell层的输出
    #         for i in range(self.n_step):
    #             (output_temp, state) = stacked_lstm(input_lstm[i], state)  # 按照顺序向stacked_lstm输入数据
    #             output.append(output_temp)
    #
    #     # 4.输出全连接层；因为LSTM每个时刻需要预测的数据维度是不一样的，为了达到要求，特在LSTM_cell层的后面加上这层进行数据转换，以满足需要
    #     with tf.name_scope('layer_41'):
    #         # X_in = X*W + b
    #         weights_out = []  # LSTM_cell层与输入层之间的连接权重
    #         biases_out = []  # 输入层神经元的偏置参数
    #         self.y = []  # 输出数据
    #         for i in range(self.n_step):
    #             weights_out_temp = tf.Variable(tf.random_uniform([Config.n_hidden_units, 1], -0.2, 0.2),
    #                                            trainable=True, name="weights_out1_" + str(i + 1))
    #             weights_out.append(weights_out_temp)
    #             biases_out_temp = tf.Variable(tf.constant(0.1, shape=[1]), trainable=True,
    #                                           name="biases_out1_" + str(i + 1))
    #             biases_out.append(biases_out_temp)
    #             y_temp = tf.nn.bias_add(tf.matmul(output[i], weights_out[i]), biases_out[i])
    #             #y_temp = tf.nn.sigmoid(y_temp, name='act_prob1_' + str(i + 1))  # 激励函数 softmax 出概率
    #             self.y.append(y_temp)
    #         self.y=tf.reduce_mean(self.y,0)
    #
    #     with tf.variable_scope('squared_TD_error1'):
    #         self.td_error = self.r + self.GAMMA * self.y_ - self.y
    #         self.loss = tf.square(tf.reduce_mean(self.td_error))    # TD_error = (r+gamma*V_next) - V_eval
    #     with tf.variable_scope('train1'):
    #         self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #
    # def learn(self,sess, x, r, x_,k):
    #     feed_dict = {}  # self.train_op更新过程中传递的参数
    #     for j in range(self.n_step):
    #         feed_dict[self.x[j]] = x_[j]
    #     y_ = sess.run(self.y, feed_dict=feed_dict)
    #     if k==0:
    #         input_x=[]
    #         input_x.append(x)
    #         for j in range(k+1):
    #             feed_dict[self.x[j]] = input_x[j]
    #     else:
    #         for j in range(self.n_step):
    #             feed_dict[self.x[j]] = x[j]
    #     feed_dict[self.y_]=y_
    #     r=np.array(r).reshape(Config.batch_size,1)
    #     feed_dict[self.r]=r
    #     td_error, _ = sess.run([self.td_error, self.train_op],feed_dict=feed_dict)
    #     return td_error