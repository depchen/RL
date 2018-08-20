# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
# from xgboost.sklearn import XGBClassifier
from bayes_opt import BayesianOptimization

import time


def a_main_2(data_manager):
    file_name = "../validate_accuracy/params_data_gp/logs.txt"
    data_file_name = "../validate_accuracy/params_data_gp/data.txt"

    global hot_method
    global data_cv, labels_cv, data_test, labels_test
    hot_method = {"paras": [], "rewards": []}
    data_dict_file = "./data_dict/data.csv"

    data_manager = data_manager
    data_cv, labels_cv = data_manager.data_cv['data_cv'], data_manager.data_cv['labels_cv']
    data_test, labels_test = data_manager.data_cv['data_test'], data_manager.data_cv['labels_test']
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

    # 随机森林
    def rfc(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap):
        global hot_method
        n_estimators = int(n_estimators) - 1
        max_depth = int(max_depth) - 1
        min_samples_split = int(min_samples_split) - 1
        min_samples_leaf = int(min_samples_leaf) - 1
        max_features = int(max_features) - 1
        bootstrap = int(bootstrap) - 1
        agr_params = [n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap]
        agr_params = np.array(agr_params).reshape(1, len(agr_params))
        val = 0
        if np.any(str(agr_params[0]) in hot_method["paras"]):
            index = hot_method["paras"].index(str(agr_params[0]))
            val = hot_method["rewards"][index]
        else:
            val = (cross_val_score(RFC(n_estimators=space_rfc['n_estimators'][n_estimators],
                                       max_depth=space_rfc['max_depth'][max_depth],
                                       min_samples_split=space_rfc['min_samples_split'][min_samples_split],
                                       min_samples_leaf=space_rfc['min_samples_leaf'][min_samples_leaf],
                                       max_features=space_rfc['max_features'][max_features],
                                       bootstrap=space_rfc['bootstrap'][bootstrap],
                                       n_jobs=-1),
                                   data_cv, labels_cv, cv=2, n_jobs=1)).mean()
            hot_method["paras"].append(str(agr_params[0]))
            hot_method["rewards"].append(val)
        global times, start_time
        time_p = time.time()
        times.append(time_p-start_time)
        return val

    gp_params = {"alpha": 1e-5}
    space_rfc = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],  # 12
        'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, None],  # 11
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],  # 21
        'min_samples_leaf': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],  # 21
        'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 9
        'bootstrap': [True, False]  # 2
    }
    # ------------huifu lishi shuju---------------;
    restore_data_dict()
    start_time = time.time()
    with open(file_name, 'a') as f:
        f.write("start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
    rfcBO = BayesianOptimization(rfc, {'n_estimators': (1, 12.9999), 'max_depth': (1, 11.9999), 'min_samples_split': (1, 21.9999),
                                       'min_samples_leaf': (1, 21.9999), 'max_features': (1, 9.9999), 'bootstrap': (1, 2.9999)},
                                 verbose=1)
    rfcBO.maximize(n_iter=300, **gp_params)
    params = pd.DataFrame(rfcBO.res['all']['params'])
    params["time"] = times[5:]
    params["accuracy"] = rfcBO.res['all']['values']
    params.to_csv(data_file_name, index=False)

    def print_test_accuracy(args, data_cv, labels_cv, data_test, labels_test):
        rfc = RFC(n_estimators=space_rfc['n_estimators'][int(args['n_estimators']) - 1],
                  max_depth=space_rfc['max_depth'][int(args['max_depth']) - 1],
                  min_samples_split=space_rfc['min_samples_split'][int(args['min_samples_split']) - 1],
                  min_samples_leaf=space_rfc['min_samples_leaf'][int(args['min_samples_leaf']) - 1],
                  max_features=space_rfc['max_features'][int(args['max_features']) - 1],
                  bootstrap=space_rfc['bootstrap'][int(args['bootstrap']) - 1],
                  n_jobs=-1)
        rfc.fit(data_cv, labels_cv)
        val = rfc.score(data_test, labels_test)
        return val

    test_accuracy = print_test_accuracy(rfcBO.res['max']['max_params'], data_cv, labels_cv, data_test, labels_test)
    with open(file_name, 'a') as f:
        f.write("\n params=\n " + str(params))
    with open(file_name, 'a') as f:
        f.write("\n best_action_param= " + str(rfcBO.res['max']['max_params']) + "\n best_action_accuracy= " + str(
            rfcBO.res['max']['max_val']) + "\n test_accuracy= " + str(test_accuracy))
    over_time = time.time()
    sum_time = over_time - start_time
    with open(file_name, 'a') as f:
        f.write("\n finish ---- search hyperParams of the algorithm ," + "start_time= " + str(start_time) +
                ", over_time= " + str(over_time) + ", sum_time = " + str(sum_time) + "\n")
    # -------------------baocun lishi shuju-----------------------;
    save_data_dict(hot_method)

    print('Final Results')
    print('RFC:', rfcBO.res['max']['max_val'])
    print('RFC:', rfcBO.res['max']['max_params'])
    # print('RFC:', rfcBO.res['all'])
    print('RFC, test_accuracy=', test_accuracy)
    print("----------GP 算法运行结束！----------")
    del data_cv, labels_cv, data_test, labels_test
    del params

if __name__ == "__main__":
    a_main_2()
