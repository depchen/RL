# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as skdata
import RL.AutoDataAnalyst_part1_compare_rfc_preSpace.code.read_data as RD
import pickle

dataset = [skdata.load_iris, skdata.load_digits, skdata.load_breast_cancer, RD.load_Adult, RD.load_Mnist, RD.load_Car_Evaluation, RD.load_litter_recognition_data_set]
X, y = dataset[6](return_X_y=True)

# y = df["attack_type"].values  # 标签，y值
# X = df[selected_feat_names].values  # 所有特征值

rfc = RandomForestClassifier(n_jobs=-1)  # 随机森林分类器

parameters = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],  # 12
    'max_depth': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, None],  # 11
    'min_samples_split': [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],  # 21
    'min_samples_leaf': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],  # 21
    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # 9
    'bootstrap': [True, False]  # 2
}



if __name__ == '__main__':
    gscv = GridSearchCV(rfc, parameters,
                        cv=3,
                        verbose=2,
                        refit=False,
                        n_jobs=1,
                        return_train_score=False)
    gscv.fit(X, y)
    print(gscv.sgscv.cv_results_)
    print(gscv.best_params_, gscv.best_score_)
    print("grid search finished")
