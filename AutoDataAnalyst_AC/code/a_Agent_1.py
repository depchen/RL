# coding:utf-8
import tensorflow as tf
import numpy as np
import time

from configFile.AgentConfigFile import AgentConfig as Config


# 类的功能：Agent.py文件的核心类，用于Agent的初始化和参数更新等；
# 拥有函数： __init__(): 构建LSTM类,并初始化；
#          getArgParams(): 获取机器学习模型的参数配置数据，用于构建机器学习模型和LSTM参数更新；
#          learn(): 学习更新LSTM的参数；
class LSTM:
    # 构建LSTM类,并初始化；
    # 输入参数：params: 机器学习算法所要搜索的超参数值的范围；
    def __init__(self, params):
        # 不添加下面这行代码，构建另一个LSTM模型结构的时候会报错！！！
        self.top_rewards = list(np.zeros([Config.batch_size]))
        self.top_agr_params = list(np.zeros([Config.batch_size]))
        tf.reset_default_graph()
        self.n_step = len(params)   # 该算法需要优化的超参数的个数
        print("self.n_step = ", self.n_step)
        # 1.输入层
        with tf.name_scope('input_layer'):
            self.x = []                                                                        # 输入数据
            self.labels = []                                                                   # 输出数据对应的标签
            self.reward = tf.placeholder(tf.float32, [Config.batch_size, 1], name='reward')    # 奖励值
            self.baseline = tf.placeholder(tf.float32, [], name='baseline')                    # 奖励基准值
            for i in range(self.n_step):
                if i == 0:
                    x_temp = tf.placeholder(tf.float32, [Config.batch_size, params[-1]], name='input_'+str(i+1))
                    self.x.append(x_temp)
                else:
                    x_temp = tf.placeholder(tf.float32, [Config.batch_size, params[i-1]], name='input_'+str(i+1))
                    self.x.append(x_temp)
                labels_temp = tf.placeholder(tf.int32, [Config.batch_size, 1], name='labels_'+str(i+1))
                self.labels.append(labels_temp)

        # 2.输入全连接层；因为LSTM每个时刻输入数据的维度不一致，添加此层的主要目的是使输入到LSTM_cell层的数据维度统一
        with tf.name_scope('layer_1'):
            # X_in = X*W + b 输入输出断的参数，要不要设置为一样的值？？？
            weights_in = []         # 输入层与LSTM_cell层之间的连接权重
            biases_in = []          # LSTM_cell层神经元的偏置参数
            for i in range(self.n_step):
                if i == 0:
                    weights_in_temp = tf.Variable(tf.random_uniform([params[-1], Config.n_hidden_units], -0.2, 0.2), trainable=True, name="weights_in_"+str(i+1))
                    weights_in.append(weights_in_temp)
                else:
                    weights_in_temp = tf.Variable(tf.random_uniform([params[i-1], Config.n_hidden_units], -0.2, 0.2), trainable=True, name="weights_in_"+str(i+1))
                    weights_in.append(weights_in_temp)
                biases_in_temp = tf.Variable(tf.constant(0.1, shape=[Config.n_hidden_units]), trainable=True, name="biases_in_"+str(i+1))
                biases_in.append(biases_in_temp)
            input_lstm = []         # LSTM_cell层的输入
            for i in range(self.n_step):
                input_lstm_temp = tf.nn.bias_add(tf.matmul(self.x[i], weights_in[i]), biases_in[i])
                input_lstm.append(input_lstm_temp)

        # 3.循环神经网络层（核心层）
        with tf.name_scope('LSTM_cell'):
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(Config.n_hidden_units, forget_bias=1.0, state_is_tuple=True) for _ in range(Config.n_layers)])
            state = stacked_lstm.zero_state(Config.batch_size, tf.float32)
            output = []             # LSTM_cell层的输出
            for i in range(self.n_step):
                (output_temp, state) = stacked_lstm(input_lstm[i], state)  # 按照顺序向stacked_lstm输入数据
                output.append(output_temp)

        # 4.输出全连接层；因为LSTM每个时刻需要预测的数据维度是不一样的，为了达到要求，特在LSTM_cell层的后面加上这层进行数据转换，以满足需要
        with tf.name_scope('layer_4'):
            # X_in = X*W + b
            weights_out = []        # LSTM_cell层与输入层之间的连接权重
            biases_out = []         # 输入层神经元的偏置参数
            self.y = []             # 输出数据
            for i in range(self.n_step):
                weights_out_temp = tf.Variable(tf.random_uniform([Config.n_hidden_units, params[i]], -0.2, 0.2), trainable=True, name="weights_out_"+str(i+1))
                weights_out.append(weights_out_temp)
                biases_out_temp = tf.Variable(tf.constant(0.1, shape=[params[i]]), trainable=True, name="biases_out_"+str(i+1))
                biases_out.append(biases_out_temp)
                y_temp = tf.nn.bias_add(tf.matmul(output[i], weights_out[i]), biases_out[i])
                y_temp = tf.nn.softmax(y_temp, name='act_prob_'+str(i+1))  # 激励函数 softmax 出概率
                self.y.append(y_temp)

        # 模型代价值 loss
        with tf.name_scope('loss'):
            nlp = []
            for i in range(self.n_step):
                nlp_temp = tf.reduce_sum(-tf.log(self.y[i]) * tf.one_hot(self.labels[i], params[i]), axis=1)
                nlp.append(nlp_temp)
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.concat(nlp, axis=1), axis=1) * (self.reward - self.baseline))
            tf.summary.scalar('loss', self.loss)
        # 模型更新操作 train
        with tf.name_scope('train'):
            # global_step = tf.Variable(0, trainable=False)
            # learning_rate = tf.train.exponential_decay(Config.lr, global_step=global_step, decay_steps=200, decay_rate=0.96, staircase=True)
            # learning_rate = tf.train.exponential_decay(Config.lr, global_step=global_step, decay_steps=400, decay_rate=0.96, staircase=True)
            # learning_rate = tf.train.exponential_decay(Config.lr, global_step=global_step, decay_steps=500, decay_rate=0.96, staircase=True)
            # self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
            self.train_op = tf.train.AdamOptimizer(Config.lr).minimize(self.loss)

    # 函数功能：获取机器学习模型参数配置数据；
    # 输入参数：sess: Session()会话对象；
    # 输出参数：x： 输入数据，用于LSTM参数更新；
    #         agr_params： 所选中的参数对应的索引位置，用于构建机器学习模型和LSTM参数更新；
    def getArgParams(self, sess, file_name, init_input_c):
        x = []                      # 输入数据 x
        agr_params = []             # 所选中的参数对应的索引位置
        init_input = init_input_c
        for i in range(self.n_step):
            x.append(init_input)
            feed_dict = {}
            for j in range(i+1):
                feed_dict[self.x[j]] = x[j]
            output_temp = sess.run(self.y[i], feed_dict=feed_dict)
            # 针对输出选择对应的动作
            with open(file_name, 'a') as f:
                f.write("output_temp[" + str(i) + "]: \n" + str(output_temp) + "\n ")
            agr_params_temp, init_input = self.getdata(output_temp)  # agr_params_temp:标签数据[batch_size, 1]; init_input:下一时刻的输入数据 [batch_size, params[i]]
            agr_params.append(agr_params_temp)
        agr_params = (np.array(agr_params).reshape(self.n_step, Config.batch_size)).T
        return x, agr_params, init_input

    # 函数功能：学习更新LSTM的参数；
    # 输入参数：sess: Session()会话对象
    #         x： 输入数据，用于LSTM参数更新； [batch_size, params]
    #         labels： 每一时刻选取参数的索引位置
    #         rewards: 每个算法结构训练完成后，在验证数据集上得到的准确率
    #         baseline_reward： 奖励基准值
    #         agr_params： 所选中的参数对应的索引位置，用于构建机器学习模型和LSTM参数更新；
    # 输入参数：loss: LSTM更新后得到的代价函数值；
    def learn(self, sess, x, labels, rewards, baseline_reward, j, writer):
        feed_dict = {}              # self.train_op更新过程中传递的参数
        for j in range(self.n_step):
            feed_dict[self.x[j]] = x[j]
            feed_dict[self.labels[j]] = (np.array(labels[:, j])).reshape(Config.batch_size, 1)
        feed_dict[self.reward] = (np.array(rewards)).reshape(Config.batch_size, 1)
        feed_dict[self.baseline] = baseline_reward
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def check_topData(self, agr_params, rewards):
        for i in range(Config.batch_size):
            for j in range(Config.batch_size):
                if rewards[i] > self.top_rewards[j]:
                    self.top_rewards[j+1:] = self.top_rewards[j:-1]
                    self.top_rewards[j] = rewards[i]
                    self.top_agr_params[j+1:] = self.top_agr_params[j:-1]
                    self.top_agr_params[j] = agr_params[i]
                    break

    def getInput(self, kk, init_input_c, c_config):
        x = []  # 输入数据 x
        top_rewards = self.top_rewards
        top_agr_params = self.top_agr_params
        top_rewards = top_rewards
        top_agr_params = np.array(top_agr_params).reshape(Config.batch_size, self.n_step)
        init_input = init_input_c
        for i in range(self.n_step):
            x.append(init_input)
            init_input = []
            for j in range(Config.batch_size):
                var_p = np.zeros(c_config.config_list[kk][i])
                var_p[top_agr_params[:, i][j]] = 1
                init_input.append(var_p)
            init_input = np.array(init_input).reshape(Config.batch_size, c_config.config_list[kk][i])

        return x, top_agr_params, top_rewards

    # 一轮更新后，初始输入值不再为1，改为上一轮迭代最后一个时刻的输出。
    # 接下来应该测试，1.batch_size的大小是否影响更新速度和准确度；2.能否识别多个最优值的情况；
    # 函数功能：将输出层的输出数据output转换成对应的标签labels和one-hot值；
    # 输入参数：output: 某一时刻，输出全连接层的输出值，数据格式为[batch_size, params[i]], 其中params[i]为第i时刻输出的维度大小；
    # 输出参数：agr_params： 当前时刻，所选中的参数对应的索引位置，用于LSTM参数更新；
    #         init_input： agr_params值的one-hot类型表示，作为下一时刻的输入；
    def getdata(self, output):
        # agr_params:标签数据
        # init_input:下一时刻的输入数据
        (p1, p2) = output.shape
        agr_params = []
        params = []
        for k in range(p1):
            p = np.random.choice(p2, 1, p=output[k])
            agr_params.append(p)
            var_p = np.zeros(p2)
            var_p[p] = 1
            params.append(var_p)
        agr_params = np.reshape(np.array(agr_params), [p1, 1])
        init_input = np.reshape(np.array(params), [p1, p2])
        return agr_params, init_input


# 虚拟环境，获取奖励值reward
def virEnv(agr_params):
    batch_size, n_step = agr_params.shape
    # [2, 3, 0, 5, 1] [2, 3, 3, 5, 1] 奖励值最大
    rewards = []
    for i in range(batch_size):
        reward_temp = 0
        if agr_params[i, 0] == 2:
            reward_temp += 0.2
        if agr_params[i, 1] == 3:
            reward_temp += 0.2
        if agr_params[i, 2] == 3:
            reward_temp += 0.2
        if agr_params[i, 2] == 0:
            reward_temp += 0.2
        if agr_params[i, 3] == 2:
            reward_temp += 0.2
        if agr_params[i, 4] == 1:
            reward_temp += 0.2
        # if np.all(agr_params[i, :] == [2, 3, 0, 5, 1]):
        #     reward_temp = 1
        rewards.append(reward_temp)
    return np.array(rewards).reshape(batch_size, 1)


# 测试主函数
def main():
    # 测试
    print("testing beginning!")
    params = [5, 6, 6, 6, 3]
    for i in range(1):
        # 模拟环境训练100轮
        print('test---begin')
        # Env.reset()
        agent = LSTM(params)
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/algorithm_" + str(i) + "/", sess.graph)
            sess.run(tf.global_variables_initializer())
            baseline_reward = 0
            start_time = time.time()
            file_name = "params_data/" + 'data_' + str(i) + '.txt'
            logs_file_name = "params_data/" + 'logs_' + str(i) + '.txt'
            with open(file_name, 'a') as f:
                f.write("\n start ---- search hyperParams of the " + str(i) + "th algorithm , start_time= " + str(
                    start_time))
            with open(logs_file_name, 'a') as f:
                f.write("\n start ---- search hyperParams of the " + str(i) + "th algorithm , start_time= " + str(
                    start_time))
            init_input = np.ones((Config.batch_size, params[-1]), np.float32)
            for j in range(200):
                print('test---middle')
                with open(file_name, 'a') as f:
                    f.write("\n i = " + str(i) + ", j = " + str(j) + "\n")
                x, agr_params, init_input = agent.getArgParams(sess, file_name, init_input)
                rewards = virEnv(agr_params)
                # rewards = np.ones([Config.batch_size, 1], int)
                if j == 0:
                    baseline_reward = np.mean(rewards)
                print("i=", i, " j=", j, " baseline_reward=", baseline_reward)
                loss = agent.learn(sess, x, agr_params, rewards, baseline_reward, j, writer)
                reward_c = np.mean(rewards)
                print("i=", i, " j=", j, "average_reward=", reward_c, " baseline_reward=", baseline_reward, " loss=",
                      loss)
                l = len(rewards)
                with open(file_name, 'a') as f:
                    f.write(
                        "agr_params = \n" + str(agr_params) + "\n rewards : " + str(rewards.reshape(1, l)) + "\n average_reward = "
                        + str(reward_c) + ", baseline_reward = " + str(baseline_reward) + ", loss = " + str(
                            loss) + "\n")
                with open(logs_file_name, 'a') as f:
                    f.write("\n i = " + str(i) + ", j = " + str(j) + "\n agr_params = \n" + str(agr_params)
                            + "\n rewards : " + str(rewards.reshape(1, l)) + "\n average_reward = " + str(
                        reward_c) + ", baseline_reward = " + str(baseline_reward) + ", loss = " + str(loss) + "\n")
                # if j % 20 == 0:
                #     print("j=", j, ", loss:", loss)
                baseline_reward = baseline_reward * 0.95 + 0.05 * (np.mean(rewards))
            over_time = time.time()
            sum_time = over_time - start_time
            with open(file_name, 'a') as f:
                f.write("\n finish ---- search hyperParams of the " + str(i) + "th algorithm ," + "start_time= " + str(
                    start_time) +
                        ", over_time= " + str(start_time) + ", sum_time = " + str(sum_time) + "\n")
            with open(logs_file_name, 'a') as f:
                f.write("\n finish ---- search hyperParams of the " + str(i) + "th algorithm ," + "start_time= " + str(
                    start_time) +
                        ", over_time= " + str(start_time) + ", sum_time = " + str(sum_time) + "\n")
            print("Cost_time:", sum_time)
    print("testing over!")

if __name__ == '__main__':
        main()


