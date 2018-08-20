# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
# from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost.sklearn import XGBClassifier
from bayes_opt import BayesianOptimization

import time


def t_main_2(data_manager,file_name,data_file_name,plot_data_path,data_dict_file):
    plot_data={"time":[],"reward":[]}
    global hot_method
    global data_cv, labels_cv, data_test, labels_test
    global a
    a=0
    hot_method = {"paras": [], "rewards": []}

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
    def rfc(max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree,
            colsample_bylevel, reg_alpha, reg_lambda):
        global hot_method
        max_depth = int(max_depth) - 1
        learning_rate = int(learning_rate) - 1
        n_estimators = int(n_estimators) - 1
        gamma = int(gamma) - 1
        min_child_weight = int(min_child_weight) - 1
        subsample = int(subsample) - 1
        colsample_bytree = int(colsample_bytree) - 1
        colsample_bylevel = int(colsample_bylevel) - 1
        reg_alpha = int(reg_alpha) - 1
        reg_lambda = int(reg_lambda) - 1
        agr_params = [max_depth, learning_rate, n_estimators, gamma, min_child_weight, subsample, colsample_bytree,
                      colsample_bylevel, reg_alpha, reg_lambda]
        agr_params = np.array(agr_params).reshape(1, len(agr_params))
        val = 0
        global a
        if np.any(str(agr_params[0]) in hot_method["paras"]):
            index = hot_method["paras"].index(str(agr_params[0]))
            val = hot_method["rewards"][index]
        else:
            val = (cross_val_score(XGBClassifier(max_depth=space_rfc['max_depth'][max_depth],
                                                 learning_rate=space_rfc['learning_rate'][learning_rate],
                                                 n_estimators=space_rfc['n_estimators'][n_estimators],
                                                 gamma=space_rfc['gamma'][gamma],
                                                 min_child_weight=space_rfc['min_child_weight'][min_child_weight],
                                                 subsample=space_rfc['subsample'][subsample],
                                                 colsample_bytree=space_rfc['colsample_bytree'][colsample_bytree],
                                                 colsample_bylevel=space_rfc['colsample_bylevel'][
                                                     colsample_bylevel],
                                                 reg_alpha=space_rfc['reg_alpha'][reg_alpha],
                                                 reg_lambda=space_rfc['reg_lambda'][reg_lambda],
                                                 nthread=-1),
                                   data_cv, labels_cv, cv=2, n_jobs=1)).mean()
            hot_method["paras"].append(str(agr_params[0]))
            hot_method["rewards"].append(val)
        global times, start_time
        time_p = time.time()
        times.append(time_p - start_time)

        plot_data["time"].append(time_p-start_time)
        plot_data["reward"].append(val)
        a=a+1
        if a%50==0:
            data=pd.DataFrame(plot_data)
            data.to_csv(plot_data_path, index=False)
        return val

    gp_params = {"alpha": 1e-5}
    space_rfc = {
        'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],  # 12
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
    # ------------huifu lishi shuju---------------;
    # restore_data_dict()
    start_time = time.time()
    with open(file_name, 'a') as f:
        f.write("start ---- search hyperParams of the algorithm , start_time= " + str(start_time))
    rfcBO = BayesianOptimization(rfc,
                                 {'max_depth': (1, 12.9999), 'learning_rate': (1, 6.9999), 'n_estimators': (1, 13.9999),
                                  'gamma': (1, 8.9999), 'min_child_weight': (1, 4.9999), 'subsample': (1, 5.9999),
                                  'colsample_bytree': (1, 6.9999), 'colsample_bylevel': (1, 6.9999),
                                  'reg_alpha': (1, 4.9999), 'reg_lambda': (1, 5.9999)},
                                 verbose=1)
    rfcBO.maximize(n_iter=1000, **gp_params)
    params = pd.DataFrame(rfcBO.res['all']['params'])
    params["time"] = times[5:]
    params["accuracy"] = rfcBO.res['all']['values']
    params.to_csv(data_file_name, index=False)

    def print_test_accuracy(args, data_cv, labels_cv, data_test, labels_test):
        rfc = XGBClassifier(max_depth=space_rfc['max_depth'][int(args['max_depth']) - 1],
                            learning_rate=space_rfc['learning_rate'][int(args['learning_rate']) - 1],
                            n_estimators=space_rfc['n_estimators'][int(args['n_estimators']) - 1],
                            gamma=space_rfc['gamma'][int(args['gamma']) - 1],
                            min_child_weight=space_rfc['min_child_weight'][int(args['min_child_weight']) - 1],
                            subsample=space_rfc['subsample'][int(args['subsample']) - 1],
                            colsample_bytree=space_rfc['colsample_bytree'][int(args['colsample_bytree']) - 1],
                            colsample_bylevel=space_rfc['colsample_bylevel'][int(args['colsample_bylevel']) - 1],
                            reg_alpha=space_rfc['reg_alpha'][int(args['reg_alpha']) - 1],
                            reg_lambda=space_rfc['reg_lambda'][int(args['reg_lambda']) - 1],
                            nthread=-1)
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
