# -*- coding: utf-8 -*-
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.DataManager import DataManager
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.t_main_1 import t_main_1
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.t_main_no_guidance import t_main_no_guidance
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.t_main_no_pre import t_main_no_pre
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.t_compared_test_gp import t_main_2
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.t_compared_test_tpe import t_main_3
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.t_compared_test_rand import t_main_4
from RL.AutoDataAnalyst_part1_compare_xgb_preSpace.code.DataManager import DataManager
import os
import warnings

print("1. 读取数据! ---------------")
data_manager_digits = DataManager(data_set_index=4)
data_manager_litter = DataManager(data_set_index=6)
warnings.filterwarnings(action = 'ignore',category = DeprecationWarning)
for i in range(10):
    print("加引导和预测网络 加引导 加预测  在手写数字和手写字母数据集的对比")
    #加引导和预测网络  手写数字
    os.mkdir("../validate_time/params_data_agent(chen)/digits_pre_net_"+str(i))
    os.mkdir("./data_dict/digits_pre_net_" + str(i))
    path="digits_pre_net_"+str(i)
    file_name = "../validate_time/params_data_agent(chen)/"+path+"/data_params.txt"
    logs_file_name = "../validate_time/params_data_agent(chen)/"+path+"/logs.txt"
    data_file_name = "../validate_time/params_data_agent(chen)/"+path+"/data.txt"
    plot_time_reward="../validate_time/params_data_agent(chen)/"+path+"/plot_time_data.csv"
    filename_p = "./data_dict/"+path+"/t_agent_1.csv"
    t_main_1(data_manager_digits,file_name,logs_file_name,data_file_name,plot_time_reward,filename_p)
    # #agent加引导  手写数字
    os.mkdir("../validate_time/params_data_agent(chen)/digits_no_pre_net_" + str(i))
    os.mkdir("./data_dict/digits_no_pre_net_" + str(i))
    path = "digits_no_pre_net_" + str(i)
    file_name = "../validate_time/params_data_agent(chen)/" + path + "/data_params.txt"
    logs_file_name = "../validate_time/params_data_agent(chen)/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_agent(chen)/" + path + "/data.txt"
    plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
    filename_p = "./data_dict/" + path + "/t_agent_1.csv"
    t_main_no_pre(data_manager_digits,file_name,logs_file_name,data_file_name,plot_time_reward,filename_p)
    # #agent加预测 手写数字
    os.mkdir("../validate_time/params_data_agent(chen)/digits_no_guidance_" + str(i))
    os.mkdir("./data_dict/digits_no_guidance_" + str(i))
    path = "digits_no_guidance_" + str(i)
    file_name = "../validate_time/params_data_agent(chen)/" + path + "/data_params.txt"
    logs_file_name = "../validate_time/params_data_agent(chen)/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_agent(chen)/" + path + "/data.txt"
    plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
    filename_p = "./data_dict/" + path + "/t_agent_1.csv"
    t_main_no_guidance(data_manager_digits, file_name, logs_file_name, data_file_name, plot_time_reward, filename_p)
    #基于贝叶斯优化 手写数字
    os.mkdir("../validate_time/params_data_gp/digits_gp_" + str(i))
    os.mkdir("./data_dict/digits_gp_" + str(i))
    path = "digits_gp_" + str(i)
    file_name = "../validate_time/params_data_gp/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_gp/" + path + "/data.txt"
    plot_data_path = "../validate_time/params_data_gp/" + path + "/plot_data.csv"
    data_dict_file = "./data_dict/" + path + "/t_gp.csv"
    t_main_2(data_manager_digits, file_name, data_file_name, plot_data_path, data_dict_file)
    # 基于TPE 手写数字
    os.mkdir("../validate_time/params_data_tpe/digits_tpe_" + str(i))
    os.mkdir("./data_dict/digits_tpe_" + str(i))
    path = "digits_tpe_" + str(i)
    file_name = "../validate_time/params_data_tpe/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_tpe/" + path + "/data.txt"
    plot_data_path = "../validate_time/params_data_tpe/" + path + "/plot_data.csv"
    data_dict_file = "./data_dict/" + path + "/t_tpe.csv"
    t_main_3(data_manager_digits, file_name, data_file_name, plot_data_path, data_dict_file)
    # 基于rand随机 手写数字
    os.mkdir("../validate_time/params_data_rand/digits_rand_" + str(i))
    os.mkdir("./data_dict/digits_rand_" + str(i))
    path = "digits_rand_" + str(i)
    file_name = "../validate_time/params_data_rand/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_rand/" + path + "/data.txt"
    plot_data_path = "../validate_time/params_data_rand/" + path + "/plot_data.csv"
    data_dict_file = "./data_dict/" + path + "/t_rand.csv"
    t_main_4(data_manager_digits, file_name, data_file_name, plot_data_path, data_dict_file)

    # 加引导和预测网络  手写字母
    os.mkdir("../validate_time/params_data_agent(chen)/litter_pre_net_" + str(i))
    os.mkdir("./data_dict/litter_pre_net_" + str(i))
    path = "litter_pre_net_" + str(i)
    file_name = "../validate_time/params_data_agent(chen)/" + path + "/data_params.txt"
    logs_file_name = "../validate_time/params_data_agent(chen)/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_agent(chen)/" + path + "/data.txt"
    plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
    filename_p = "./data_dict/" + path + "/t_agent_1.csv"
    t_main_1(data_manager_litter, file_name, logs_file_name, data_file_name, plot_time_reward, filename_p)
    # agent加引导  手写字母
    os.mkdir("../validate_time/params_data_agent(chen)/litter_no_pre_net_" + str(i))
    os.mkdir("./data_dict/litter_no_pre_net_" + str(i))
    path = "litter_no_pre_net_" + str(i)
    file_name = "../validate_time/params_data_agent(chen)/" + path + "/data_params.txt"
    logs_file_name = "../validate_time/params_data_agent(chen)/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_agent(chen)/" + path + "/data.txt"
    plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
    filename_p = "./data_dict/" + path + "/t_agent_1.csv"
    t_main_no_pre(data_manager_litter, file_name, logs_file_name, data_file_name, plot_time_reward, filename_p)
    # agent加预测 手写字母
    os.mkdir("../validate_time/params_data_agent(chen)/litter_no_guidance_" + str(i))
    os.mkdir("./data_dict/litter_no_guidance_" + str(i))
    path = "litter_no_guidance_" + str(i)
    file_name = "../validate_time/params_data_agent(chen)/" + path + "/data_params.txt"
    logs_file_name = "../validate_time/params_data_agent(chen)/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_agent(chen)/" + path + "/data.txt"
    plot_time_reward = "../validate_time/params_data_agent(chen)/" + path + "/plot_time_data.csv"
    filename_p = "./data_dict/" + path + "/t_agent_1.csv"
    t_main_no_guidance(data_manager_litter, file_name, logs_file_name, data_file_name, plot_time_reward, filename_p)
    #基于贝叶斯优化 手写字母
    os.mkdir("../validate_time/params_data_gp/litter_gp_" + str(i))
    os.mkdir("./data_dict/litter_gp_" + str(i))
    path = "litter_gp_" + str(i)
    file_name = "../validate_time/params_data_gp/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_gp/" + path + "/data.txt"
    plot_data_path = "../validate_time/params_data_gp/" + path + "/plot_data.csv"
    data_dict_file = "./data_dict/" + path + "/t_gp.csv"
    t_main_2(data_manager_litter, file_name, data_file_name, plot_data_path, data_dict_file)
    # 基于TPE 手写字母
    os.mkdir("../validate_time/params_data_tpe/litter_tpe_" + str(i))
    os.mkdir("./data_dict/litter_tpe_" + str(i))
    path = "litter_tpe_" + str(i)
    file_name = "../validate_time/params_data_tpe/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_tpe/" + path + "/data.txt"
    plot_data_path = "../validate_time/params_data_tpe/" + path + "/plot_data.csv"
    data_dict_file = "./data_dict/" + path + "/t_tpe.csv"
    t_main_3(data_manager_litter, file_name, data_file_name, plot_data_path, data_dict_file)
    # 基于rand随机 手写字母
    os.mkdir("../validate_time/params_data_rand/litter_rand_" + str(i))
    os.mkdir("./data_dict/litter_rand_" + str(i))
    path = "litter_rand_" + str(i)
    file_name = "../validate_time/params_data_rand/" + path + "/logs.txt"
    data_file_name = "../validate_time/params_data_rand/" + path + "/data.txt"
    plot_data_path = "../validate_time/params_data_rand/" + path + "/plot_data.csv"
    data_dict_file = "./data_dict/" + path + "/t_rand.csv"
    t_main_4(data_manager_litter, file_name, data_file_name, plot_data_path, data_dict_file)

