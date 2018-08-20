# coding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import time

from a_Agent_1 import LSTM
from EnvironmentManager import EnvironmentManager
from configFile.MainConfigureFile import MainConfig
from configFile.AgentConfigFile import AgentConfig


# 主函数
def a_main_1(data_manager):
    data_manager = data_manager
    envManager = EnvironmentManager(data_manager)
    envManager.auto_create_multi_singleprocess_envs()

    for i in range(1):
        Env, params = envManager.next_environment()
        # 恢复文件中的字典数据
        # Env.restore_data_dict()
        agent = LSTM(params)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("../logs/algorithm_" + str(i) + "/", sess.graph)
            baseline_reward = 0
            start_time = time.time()
            file_name = "../validate_accuracy/params_data_agent(chen)/data_params.txt"
            logs_file_name = "../validate_accuracy/params_data_agent(chen)/logs.txt"
            data_file_name = "../validate_accuracy/params_data_agent(chen)/data.txt"
            with open(file_name, 'a') as f:
                f.write("\n start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
            with open(logs_file_name, 'a') as f:
                f.write("\n start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
            init_input = np.ones((AgentConfig.batch_size, params[-1]), np.float32)
            params_data = None
            for j in range(MainConfig.num_train):
                with open(file_name, 'a') as f:
                    f.write("\n i = " + str(i) + ", j = " + str(j) + "\n")
                x, agr_params, _ = agent.getArgParams(sess, file_name, init_input)
                rewards = Env.run(agr_params)
                # 记录下top 级的数据,等到合适的时候再放进模型训练；
                temp_time = time.time()
                per_time = temp_time - start_time
                agr_paramss = []
                time_ts = []
                for kf in range(agr_params.shape[0]):
                    agr_params_temp = str(agr_params[kf])
                    agr_paramss.append(agr_params_temp)
                    time_ts.append(per_time)
                data_temp = np.array(agr_paramss + time_ts).reshape([2, agr_params.shape[0]]).T
                params_data_temp = pd.DataFrame(data_temp, columns=["agr_params", "time"])
                params_data_temp["accuracy"] = rewards
                if j == 0:
                    params_data = params_data_temp
                else:
                    params_data = pd.concat([params_data, params_data_temp], ignore_index=True)

                agent.check_topData(agr_params, rewards)
                if j == 0:
                    baseline_reward = np.mean(rewards)
                if (j+1) % 10 == 0:
                    x, agr_params, rewards = agent.getInput(i, init_input, envManager.c_config)
                    print("if: algorithm rectify, rewards:", rewards)
                    loss = agent.learn(sess, x, agr_params, rewards, baseline_reward, j, writer)
                else:
                    print("else: normal training, rewards:", rewards)
                    loss = agent.learn(sess, x, agr_params, rewards, baseline_reward, j, writer)
                reward_c = np.mean(rewards)
                print("i=", i, " j=", j, "average_reward=", reward_c, " baseline_reward=", baseline_reward, " loss=", loss, "\n")
                with open(file_name, 'a') as f:
                    f.write("agr_params = \n" + str(agr_params) + "\n rewards : " + str(rewards) + "\n average_reward = "
                            + str(reward_c) + ", baseline_reward = " + str(baseline_reward) + ", loss = " + str(loss) + "\n")
                with open(logs_file_name, 'a') as f:
                    f.write("\n i = " + str(i) + ", j = " + str(j) + "\n agr_params = \n" + str(agr_params)
                            + "\n rewards : " + str(rewards) + "\n average_reward = " + str(reward_c) + ", baseline_reward = " + str(baseline_reward) + ", loss = " + str(loss) + "\n")
                baseline_reward = baseline_reward * AgentConfig.dr + (1-AgentConfig.dr) * reward_c
            over_time = time.time()
            sum_time = over_time-start_time
            params_data.to_csv(data_file_name, index=False)

            test_accuracy = Env.get_test_accuracy(agent, file_name, logs_file_name)
            print('test_accuracy=', test_accuracy)

            with open(file_name, 'a') as f:
                f.write("\n finish ---- search hyperParams of the " + str(i) + "th algorithm ," + "start_time= " + str(start_time) +
                        ", over_time= " + str(over_time) + ", sum_time = " + str(sum_time) + "\n")
            with open(logs_file_name, 'a') as f:
                f.write("\n finish ---- search hyperParams of the " + str(i) + "th algorithm ," + "start_time= " + str(start_time) +
                        ", over_time= " + str(over_time) + ", sum_time = " + str(sum_time) + "\n")
            print("Cost_time:", sum_time)
        # 保存文件中的字典数据
        Env.save_data_dict()
    print("---------训练结束!----------")

if __name__ == '__main__':
        a_main_1()
