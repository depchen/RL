# coding:utf-8
import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, space_eval

import time


def t_main_3(data_manager,file_name,data_file_name,plot_data_path,data_dict_file):
    plot_data = {"time": [], "reward": []}

    global hot_method
    global data_cv, labels_cv
    global params, rewards
    global a
    a = 0
    hot_method = {"paras": [], "rewards": []}

    data_manager = data_manager
    data_cv, labels_cv = data_manager.data_cv['data_cv'], data_manager.data_cv['labels_cv']
    data_test, labels_test = data_manager.data_cv['data_test'], data_manager.data_cv['labels_test']
    params = []
    rewards = []
    global times, start_time
    times = []

    def save_data_dict(hot_method_p):
        data = pd.DataFrame(data=hot_method_p)
        data.to_csv(data_dict_file, index=False)
        data_length = len(hot_method_p["paras"])
        print("successfull !！！ save total ", data_length, " data!")
        return data_length

    def restore_data_dict():
        global hot_method
        data_dict = pd.read_csv(data_dict_file, index_col=False)
        hot_method["paras"] = list(data_dict["paras"].values)
        hot_method["rewards"] = list(data_dict["rewards"].values)
        data_length = len(hot_method["paras"])
        print("successfull !！！ restore total ", data_length, " data!")
        return data_length

    space_p = {'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],  # 12
               'learning_rate': [0.001, 0.005, 0.01, 0.04, 0.07, 0.1],  # 6
               'n_estimators': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],  # 13
               'gamma': [0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # 8
               'min_child_weight': [1, 3, 5, 7],  # 4
               'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # 5
               'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 6
               'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 6
               'reg_alpha': [0, 0.1, 0.5, 1.0],  # 4
               'reg_lambda': [0.01, 0.03, 0.07, 0.1, 1.0]  # 5
               }

    def func(args, flag=None):
        global hot_method, start_time
        global params, times, rewards
        global data_cv, labels_cv

        max_depth = space_p["max_depth"].index(args["max_depth"])
        learning_rate = space_p["learning_rate"].index(args["learning_rate"])
        n_estimators = space_p["n_estimators"].index(args["n_estimators"])
        gamma = space_p["gamma"].index(args["gamma"])
        min_child_weight = space_p["min_child_weight"].index(args["min_child_weight"])
        subsample = space_p["subsample"].index(args["subsample"])
        colsample_bytree = space_p["colsample_bytree"].index(args["colsample_bytree"])
        colsample_bylevel = space_p["colsample_bylevel"].index(args["colsample_bylevel"])
        reg_alpha = space_p["reg_alpha"].index(args["reg_alpha"])
        reg_lambda = space_p["reg_lambda"].index(args["reg_lambda"])

        agr_params = [max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree,
                      colsample_bylevel, reg_alpha, reg_lambda]
        agr_params = np.array(agr_params).reshape(1, len(agr_params))
        val = 0
        global a
        if np.any(str(agr_params[0]) in hot_method["paras"]):
            index = hot_method["paras"].index(str(agr_params[0]))
            val = hot_method["rewards"][index]
            print("if!!!")
        else:
            print("else!!!")
            rfc = XGBClassifier(max_depth=args["max_depth"],
                                learning_rate=args["learning_rate"],
                                n_estimators=args["n_estimators"],
                                gamma=args["gamma"],
                                min_child_weight=args['min_child_weight'],
                                subsample=args["subsample"],
                                colsample_bytree=args["colsample_bytree"],
                                colsample_bylevel=args["colsample_bylevel"],
                                reg_alpha=args['reg_alpha'],
                                reg_lambda=args["reg_lambda"],
                                nthread=-1)
            results = cross_val_score(rfc, data_cv, labels_cv, cv=2, n_jobs=1)
            val = np.mean(results)
            hot_method["paras"].append(str(agr_params[0]))
            hot_method["rewards"].append(val)
        if flag == None:
            params.append(args)
            time_p = time.time()
            times.append(time_p - start_time)
            rewards.append(val)
        plot_data["time"].append(time_p - start_time)
        plot_data["reward"].append(val)
        a = a + 1
        if a % 50 == 0:
            data = pd.DataFrame(plot_data)
            data.to_csv(plot_data_path, index=False)
        return -val

    space = {
        'max_depth': hp.choice("max_depth", [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]),  # 12
        'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.04, 0.07, 0.1]),  # 6
        'n_estimators': hp.choice('n_estimators', [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]), # 13
        'gamma': hp.choice('gamma', [0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),  # 8
        'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),  # 4
        'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),  # 5
        'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),  # 6
        'colsample_bylevel': hp.choice('colsample_bylevel', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),  # 6
        'reg_alpha': hp.choice('reg_alpha', [0, 0.1, 0.5, 1.0]),  # 4
        'reg_lambda': hp.choice('reg_lambda', [0.01, 0.03, 0.07, 0.1, 1.0])  # 5
    }
    # ------------huifu lishi shuju---------------;
    # restore_data_dict()
    start_time = time.time()
    with open(file_name, 'a') as f:
        f.write("start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
    best = fmin(func, space, algo=tpe.suggest, max_evals=1000)
    params = pd.DataFrame(params)
    params["time"] = times
    params["accuracy"] = rewards
    params.to_csv(data_file_name, index=False)

    def print_test_accuracy(args, data_cv, labels_cv, data_test, labels_test):
        rfc = XGBClassifier(max_depth=args["max_depth"], learning_rate=args["learning_rate"],
                            n_estimators=args["n_estimators"], gamma=args["gamma"],
                            min_child_weight=args['min_child_weight'], subsample=args["subsample"],
                            colsample_bytree=args["colsample_bytree"],
                            colsample_bylevel=args["colsample_bylevel"], reg_alpha=args['reg_alpha'],
                            reg_lambda=args["reg_lambda"],
                            nthread=-1)
        rfc.fit(data_cv, labels_cv)
        val = rfc.score(data_test, labels_test)
        return val

    test_accuracy = print_test_accuracy(space_eval(space, best), data_cv, labels_cv, data_test, labels_test)

    with open(file_name, 'a') as f:
        f.write("\n params=\n " + str(params))
    with open(file_name, 'a') as f:
        f.write("\n best_action_index= " + str(best) + "\n best_action_param= " + str(
            space_eval(space, best)) + "\n best_action_accuracy= " + str(
            func(space_eval(space, best), 1)) + "\n test_accuracy= " + str(test_accuracy))
    over_time = time.time()
    sum_time = over_time - start_time
    with open(file_name, 'a') as f:
        f.write("\n finish ---- search hyperParams of the algorithm ," + "start_time= " + str(start_time) +
                ", over_time= " + str(over_time) + ", sum_time = " + str(sum_time) + "\n")
    # -------------------baocun lishi shuju-----------------------;
    save_data_dict(hot_method)

    print("best_action_index", best)
    print("-----best_action_param:", space_eval(space, best))
    print("-----best_action_accuracy,", func(space_eval(space, best), 1))
    print('RFC, test_accuracy=', test_accuracy)
    print("----------TPE 算法运行结束！----------")
    # print("params:", params)
    del data_cv, labels_cv, data_test, labels_test
    del params, rewards, times, start_time

if __name__ == '__main__':
    t_main_3()
