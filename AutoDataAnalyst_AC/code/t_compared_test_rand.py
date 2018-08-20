# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, rand, hp, space_eval

import time


def t_main_4(data_manager):
    file_name = "../validate_time/params_data_rand/logs.txt"
    data_file_name = "../validate_time/params_data_rand/data.txt"
    data_dict_file = "./data_dict/t_rand.csv"
    plot_data_path = "../validate_time/params_data_rand/plot_data.csv"
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

    space_p = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],  # 12
               'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],  # 11
               'min_samples_split': [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], # 21
               'min_samples_leaf': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], # 21
               'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 9
               'bootstrap': [True, False]  # 2
               }

    def func(args, flag=None):
        global hot_method, start_time
        global params, times, rewards
        global data_cv, labels_cv

        n_estimators = space_p["n_estimators"].index(args["n_estimators"])
        max_depth = space_p["max_depth"].index(args["max_depth"])
        min_samples_split = space_p["min_samples_split"].index(args["min_samples_split"])
        min_samples_leaf = space_p["min_samples_leaf"].index(args["min_samples_leaf"])
        max_features = space_p["max_features"].index(args["max_features"])
        bootstrap = space_p["bootstrap"].index(args["bootstrap"])

        agr_params = [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap]
        agr_params = np.array(agr_params).reshape(1, len(agr_params))
        val = 0
        global a
        if np.any(str(agr_params[0]) in hot_method["paras"]):
            index = hot_method["paras"].index(str(agr_params[0]))
            val = hot_method["rewards"][index]
            print("if!!!")
        else:
            print("else!!!")
            rfc = RandomForestClassifier(n_estimators=args["n_estimators"],
                                         max_depth=args["max_depth"],
                                         min_samples_split=args["min_samples_split"],
                                         min_samples_leaf=args["min_samples_leaf"],
                                         max_features=args['max_features'],
                                         bootstrap=args["bootstrap"],
                                         n_jobs=-1)
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
        'n_estimators': hp.choice("n_estimators", [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]), # 12
        'max_depth': hp.choice('max_depth', [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),  # 11
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]),  # 21
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]),  # 21
        'max_features': hp.choice('max_features', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),  # 9
        'bootstrap': hp.choice('bootstrap', [True, False])  # 2
    }
    # ------------huifu lishi shuju---------------;
    # restore_data_dict()
    start_time = time.time()
    with open(file_name, 'a') as f:
        f.write("start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
    best = fmin(func, space, algo=rand.suggest, max_evals=1000)
    params = pd.DataFrame(params)
    params["time"] = times
    params["accuracy"] = rewards
    params.to_csv(data_file_name, index=False)

    def print_test_accuracy(args, data_cv, labels_cv, data_test, labels_test):
        rfc = RandomForestClassifier(n_estimators=args["n_estimators"], max_depth=args["max_depth"], min_samples_split=args["min_samples_split"],
                                     min_samples_leaf=args["min_samples_leaf"], max_features=args['max_features'], bootstrap=args["bootstrap"],
                                     n_jobs=-1)
        rfc.fit(data_cv, labels_cv)
        val = rfc.score(data_test, labels_test)
        return val
    test_accuracy = print_test_accuracy(space_eval(space, best), data_cv, labels_cv, data_test, labels_test)

    with open(file_name, 'a') as f:
        f.write("\n params=\n " + str(params))
    with open(file_name, 'a') as f:
        f.write("\n best_action_index= " + str(best) + "\n best_action_param= " + str(space_eval(space, best)) + "\n best_action_accuracy= " + str(func(space_eval(space, best), 1)) + "\n test_accuracy= " + str(test_accuracy))
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
    data_manager = DataManager()
    t_main_4(data_manager)

