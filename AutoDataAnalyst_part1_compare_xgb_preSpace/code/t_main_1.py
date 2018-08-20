# coding:utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import time

from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.t_Agent_1 import LSTM
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.EnvironmentManager import EnvironmentManager
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.configFile.MainConfigureFile import MainConfig
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.configFile.AgentConfigFile import AgentConfig
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.NNet import NNet

# 主函数
def t_main_1(data_manager,file_name,logs_file_name,data_file_name,plot_time_reward,filename_p):
    data_manager = data_manager
    envManager = EnvironmentManager(data_manager)
    envManager.auto_create_multi_singleprocess_envs()
    plot_data = {"time": [], "rewards_max": [], "rewards_mean": [], "reward_min": []}
    plot_all_data = {"time": [], "rewards": []}
    _100_real_data=[]

    for i in range(1):
        Env, params = envManager.next_environment()
        # 恢复文件中的字典数据
        # Env.restore_data_dict()
        agent = LSTM(params)
        nnet = NNet(len(params))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("../logs/algorithm_" + str(i) + "/", sess.graph)
            baseline_reward = 0
            start_time = time.time()
            #in_100_real_data="../validate_time/params_data_agent(chen)/in_100_real_data.csv"
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
                # #不使用神经网络
                # rewards = Env.run(agr_params)
                # step_time=time.time()
                # one_time=step_time-start_time
                # plot_data["time"].append(one_time)
                # plot_data["rewards"].append(np.max(rewards))
                # if j%100==0:
                #     plot = pd.DataFrame(data=plot_data)
                #     plot.to_csv(plot_time_reward, index=False)
                #使用神经网络
                if j<=25:
                    rewards = Env.run(agr_params)
                    nnet.store_transition(agr_params,rewards)
                    step_time=time.time()
                    one_time=step_time-start_time
                    plot_data["time"].append(one_time)
                    #a=np.mean(rewards)
                    plot_data["rewards_max"].append(np.max(rewards))
                    plot_data["rewards_mean"].append(np.mean(rewards))
                    plot_data["reward_min"].append(np.min(rewards))

                    train_net_reward=np.array(rewards).reshape(8,1)
                    #nnet_params=Env.getparams(agr_params)
                    nnet.train_net(sess,j)
                    #if j>=5:
                    #nnet.train_net(sess, j)
                    nnet.train_net(sess, j)
                if j>25 and j<150:
                    #nnet_params = Env.getparams(agr_params)
                    #rewards = Env.run(agr_params)
                    # _100_real_data.append(rewards)

                    rewards=nnet.get_reward(sess,agr_params)
                    rewards = np.array(rewards).reshape(AgentConfig.batch_size)
                    #step_time = time.time()
                    #one_time = step_time - start_time
                    #plot_data["time"].append(one_time)
                    #plot_data["rewards"].append(np.mean(rewards))
                if j>=150:
                    #plot = pd.DataFrame(data=plot_data)
                    #plot.to_csv(plot_time_reward, index=False)
                    # 存储中间100次的真实reward.用于比对
                    #plot = pd.DataFrame(data=_100_real_data)
                    #plot.to_csv(in_100_real_data, index=True)
                    if j%100:
                        plot = pd.DataFrame(data=plot_data)
                        plot.to_csv(plot_time_reward, index=False)
                    rewards = Env.run(agr_params)
                    step_time = time.time()
                    one_time = step_time - start_time
                    plot_data["time"].append(one_time)
                    plot_data["rewards_max"].append(np.max(rewards))
                    plot_data["rewards_mean"].append(np.mean(rewards))
                    plot_data["reward_min"].append(np.min(rewards))
                #记录所有数据
                if j<=25 or j>=150:
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
                    #设置params_data 用来讲每组超参数-对应耗费时间-对应reward 一一对应  便于其后存储cvs
                    if j == 0:
                        params_data = params_data_temp
                    else:
                        params_data = pd.concat([params_data, params_data_temp], ignore_index=True)
                # 记录下top 级的数据,等到合适的时候再放进模型训练；
                agent.check_topData(agr_params, rewards)
                if j == 0:
                    baseline_reward = np.mean(rewards)
                #每十步 讲使用引导数据池 即最好的reward的超参数值   用于减少方差
                if (j+1) % 5 == 0:
                    x1,x2, agr_params, rewards = agent.getInput(i, init_input, envManager.c_config)
                    rewards=np.array(rewards).reshape(AgentConfig.batch_size*2, 1)

                    print("if: algorithm rectify, rewards:", np.array(rewards[8:16, :]).flatten())
                    loss = agent.learn(sess, x2, agr_params[8:16, :], rewards[8:16, :], baseline_reward, j, writer)
                    reward_c=np.mean(rewards[8:16, :])
                    print("i=", i, " j=", j, "average_reward=",reward_c , " baseline_reward=", baseline_reward,
                          " loss=", loss, "\n")
                    #baseline_reward = baseline_reward * AgentConfig.dr + (1 - AgentConfig.dr) * reward_c

                    print("if: algorithm rectify, rewards:", np.array(rewards[0:8, :]).flatten())
                    loss = agent.learn(sess, x1, agr_params[0:8,:], rewards[0:8,:], baseline_reward, j, writer)
                    rewards=rewards[0:8,:]
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
            #存储300次 每次的实时reward和time
            plot = pd.DataFrame(data=plot_data)
            plot.to_csv(plot_time_reward, index=False)


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

        Env.save_data_dict(filename_p)
    print("---------训练结束!----------")
# data_manager=DataManager()
# if __name__ == '__main__':
#     warnings.filterwarnings(action = 'ignore',category = DeprecationWarning)
#     file_name = "../validate_time/params_data_agent(chen)/data_params.txt"
#     logs_file_name = "../validate_time/params_data_agent(chen)/logs.txt"
#     data_file_name = "../validate_time/params_data_agent(chen)/data.txt"
#     plot_time_reward = "../validate_time/params_data_agent(chen)/plot_time_data.csv"
#     filename_p = "./data_dict/t_agent_1.csv"
#     t_main_1(data_manager,file_name,logs_file_name,data_file_name,plot_time_reward,filename_p)
