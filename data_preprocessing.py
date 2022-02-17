import pandas as pd
import os
import glob
import time
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_tree
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import model_selection, metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV  # Perforing grid search


def read_datasets(dataset_path: str):
    all_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    cases_data = pd.concat((pd.read_csv(f, low_memory=False) for f in all_files))
    return cases_data


def preprocess_dataset(tested_individuals_dataset: pd.DataFrame):
    # removing rows with empty empty and None values
    cases_data = tested_individuals_dataset.mask(tested_individuals_dataset.eq('None')).dropna()
    # removing rows where corona result has misleading value
    cases_data = cases_data[((cases_data.corona_result == 'positive') | (cases_data.corona_result == 'negative'))]
    # replace corona_result from 'Yes' and 'No' to 1 and 0
    cases_data['corona_result'] = (cases_data['corona_result'] == 'positive').astype(int)
    # replace age_60_and_above from 'Yes' and 'No' to 1 and 0
    cases_data['age_60_and_above'] = (cases_data['age_60_and_above'] == 'Yes').astype(int)
    # replace gender from 'male' and 'female' to 1 and 0
    cases_data['gender'] = (cases_data['gender'] == 'female').astype(int)
    # replace test_indication from 'Contact with confirmed', 'Abroad', 'Other' into 0, 1 and 2
    cases_data.loc[cases_data['test_indication'] == 'Contact with confirmed', 'test_indication'] = 0
    cases_data.loc[cases_data['test_indication'] == 'Abroad', 'test_indication'] = 1
    cases_data.loc[cases_data['test_indication'] == 'Other', 'test_indication'] = 2

    # df['Embarked'].replace(('Q', 'S', 'C'), (0, 1, 2), inplace=True)
    # df['Sex'].replace(('male', 'female'), (0, 1), inplace=True)
    cases_data = cases_data.drop_duplicates()
    return cases_data


def update_dataset_structure(dataset: pd.DataFrame):
    cases_data_copy = dataset.copy()
    corona_results = cases_data_copy.pop('corona_result')
    cases_data_copy = cases_data_copy.drop(['test_date'], axis=1).astype(int)
    cases_data_copy.insert(len(cases_data_copy.columns), corona_results.name, corona_results)
    print(cases_data_copy.iloc[1])
    return cases_data_copy


def gbmModelfit(alg, x_train, y_train, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    alg.fit(x_train, y_train)

    # Predict training set:
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = model_selection.cross_val_score(alg, x_train, y_train, cv=cv_folds, scoring='roc_auc')

    # Print model report:
    print("\nModel Report")
    print("F1 : %.4g" % metrics.f1_score(y_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
            np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importance', figsize=(13, 5), edgecolor='white', linewidth=5)
        plt.xticks(fontsize=9, rotation=0)
        plt.ylabel('Feature Importance Score')
        plt.show()


def xgb_model_fit(alg, x_train, y_train):
    #, useTrainCV, cv_folds=5, early_stopping_rounds=1200)
    # if useTrainCV:
    #     xgb_param = alg.get_xgb_params()
    #     xgtrain = xgb.DMatrix(x_train.values, label=y_train.values)
    #     cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
    #                       metrics='auc', early_stopping_rounds=early_stopping_rounds)
    #     alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("F1 : %.4g" % metrics.f1_score(y_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance', figsize=(13, 5), edgecolor='white', linewidth=5)
    plt.xticks(fontsize=9, rotation=0)
    plt.ylabel('Feature Importance Score')
    plt.show()


def xgbrf_model_fit(alg, x_train, y_train):
    # Fit the algorithm on the data
    alg.fit(x_train, y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(x_train)
    dtrain_predprob = alg.predict_proba(x_train)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("F1 : %.4g" % metrics.f1_score(y_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importance', figsize=(13, 5), edgecolor='white', linewidth=5)
    plt.xticks(fontsize=9, rotation=0)
    plt.ylabel('Feature Importance Score')
    plt.show()


def main():
    pd.set_option('display.max_columns', 11)
    os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz\\bin'
    dataset_path = 'datasets/'
    # read dataset from multiple files and concatenate into one dataset
    cases_data = read_datasets(dataset_path=dataset_path)
    # process dataset data - remove columns with empty, None, or misleading values
    cases_data = preprocess_dataset(tested_individuals_dataset=cases_data)
    # print(cases_data)
    cases_data.to_csv(dataset_path + 'preprocessed_dataset/corona_tested_individuals.csv', index=False)

    # removing columns which will not be used during training
    cases_data = update_dataset_structure(dataset=cases_data)

    print(cases_data.head())

    X, y = cases_data.iloc[:, :-1], cases_data.iloc[:, -1]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=146)
    #
    # gbm = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1200,
    #                                  min_samples_leaf=100, max_depth=5,
    #                                  max_features='sqrt', loss='deviance',
    #                                  n_estimators=500)
    # start_time = time.time()
    #
    # gbmModelfit(gbm, X_train, y_train, X_train.columns)
    #
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # predictions = gbm.predict(X_test)
    #
    # print("Accuracy on test set: %.4g" % metrics.f1_score(y_test.values, predictions))




    # estimators and learning rate estimation
    # param_test1 = {'n_estimators': range(20, 81, 10)}
    #
    # gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1500,
    #                                                              min_samples_leaf=150, max_depth=8,
    #                                                              max_features='sqrt', subsample=0.8,
    #                                                              random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', n_jobs=4, cv=5)
    #
    # gsearch1.fit(X_train, y_train)
    #
    # print('results')
    # print(gsearch1.cv_results_)
    # print('params')
    # print(gsearch1.best_params_)
    # print('scores')
    # print(gsearch1.best_score_)
    #
    # # TEST parameters 2 - tree specific parameters (depth and sample split)
    # param_test2 = {'max_depth': range(5, 16, 2), 'min_samples_split': range(800, 2000, 200)}
    # gsearch2 = GridSearchCV(
    #     estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8,
    #                                          random_state=10),
    #     param_grid=param_test2, scoring='roc_auc', n_jobs=4, cv=5)
    # gsearch2.fit(X_train, y_train)
    #
    # print('results')
    # print(gsearch2.cv_results_)
    # print('params')
    # print(gsearch2.best_params_)
    # print('scores')
    # print(gsearch2.best_score_)

    # XGBOOST
    # xgb_cl = XGBClassifier(booster='gbtree', verbosity=1,
    #                        use_label_encoder=False, tree_method='gpu_hist',
    #                        learning_rate=0.05, n_estimators=5000, predictor='gpu_predictor')
    # xgb_cl.fit(X_train, y_train)
    # preds = xgb_cl.predict(X_test)
    # print("Accuracy : %.4g" % metrics.f1_score(y_test.values, preds))
    # xgb.plot_tree(xgb_cl, num_trees=2)
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(150, 100)
    # fig.savefig('tree.png')


    # print(y_train['corona_result'].value_counts())


    #XGB Tunning
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=146)
    #
    # xgb1 = XGBClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
    #                     n_estimators=20, objective='binary:logistic', booster='gbtree', n_jobs=4, eta=0.5)
    #
    # start_time = time.time()
    #
    # xgb_model_fit(xgb1, X_train, y_train)
    #
    # predictions = xgb1.predict(X_test)
    #
    # print("--- %s XGB seconds ---" % (time.time() - start_time))
    #
    # print("Accuracy on test set: %.4g" % metrics.f1_score(y_test.values, predictions))



    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=127)
    #
    # xgb = XGBClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
    #                     n_estimators=20, objective='binary:logistic', booster='gbtree', n_jobs=4, eta=0.3)
    #
    # start_time = time.time()
    #
    # xgb_model_fit(xgb, X_train, y_train)
    #
    # predictions = xgb.predict(X_test)
    #
    # print("--- %s XGB2 seconds ---" % (time.time() - start_time))
    #
    # print("F-score on test set: %.4g" % metrics.f1_score(y_test.values, predictions))
    # print("Accuracy on test set: %.4g" % metrics.accuracy_score(y_test.values, predictions))
    #
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=127)

    xgb = XGBClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
                        n_estimators=400, objective='binary:logistic', booster='gbtree', n_jobs=4, eta=0.3)

    start_time = time.time()

    xgb_model_fit(xgb, X_train, y_train)

    predictions = xgb.predict(X_test)

    print("--- %s XGB500 seconds ---" % (time.time() - start_time))

    print("F-score on test set: %.4g" % metrics.f1_score(y_test.values, predictions))
    print("Accuracy on test set: %.4g" % metrics.accuracy_score(y_test.values, predictions))



    # param_test1 = {
    #     'n_estimators': [40, 140, 500, 1000, 2700, 4800, 8400, 10000],
    #     'eta': [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
    # }
    #
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(eta=0.01, n_estimators=100, max_depth=6,
    #                                                 objective='binary:logistic', nthread=6,
    #                                                 use_label_encoder=False, predictor='gpu_predictor',
    #                                                 tree_method='gpu_hist'),
    #                         param_grid=param_test1, scoring='roc_auc', n_jobs=6, cv=5)
    #
    # gsearch1.fit(X_train, y_train)
    #
    # print('results')
    # print(gsearch1.cv_results_)
    # print('params')
    # print(gsearch1.best_params_)
    # print('scores')
    # print(gsearch1.best_score_)




    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=177)
    #
    # xgb_rf_classif = xgb.XGBRFClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
    #                                      n_estimators=20, objective='binary:logistic', booster='gbtree', n_jobs=4,
    #                                      learning_rate=0.5, max_depth=4)
    # start_time = time.time()
    # xgbrf_model_fit(xgb_rf_classif, X_train, y_train)
    # print("--- %s XGBRF1 seconds ---" % (time.time() - start_time))
    #
    # print("Test  Accuracy Score : %.2f"%xgb_rf_classif.score(X_test, y_test))
    # print("Train Accuracy Score : %.2f"%xgb_rf_classif.score(X_train, y_train))
    #
    # predictions = xgb_rf_classif.predict(X_test)
    #
    # print("F-score XGBRF on test 20  set: %.4g" % metrics.f1_score(y_test.values, predictions))
    # print("Accuracy on test set: %.4g" % metrics.accuracy_score(y_test.values, predictions))



    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=175)
    #
    # xgb_rf_classif = xgb.XGBRFClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
    #                                      n_estimators=500, objective='binary:logistic', booster='gbtree', n_jobs=4,
    #                                      learning_rate=0.05)
    # start_time = time.time()
    # xgbrf_model_fit(xgb_rf_classif, X_train, y_train)
    # print("--- %s XGBRF1 seconds ---" % (time.time() - start_time))
    #
    # print("Test  Accuracy Score : %.2f"%xgb_rf_classif.score(X_test, y_test))
    # print("Train Accuracy Score : %.2f"%xgb_rf_classif.score(X_train, y_train))
    #
    # predictions = xgb_rf_classif.predict(X_test)
    #
    # print("F-score XGBRF on test 600  set: %.4g" % metrics.f1_score(y_test.values, predictions))
    # print("Accuracy on test set: %.4g" % metrics.accuracy_score(y_test.values, predictions))

    #
    #
    #
    #
    # #XGBRFClassifier 2
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=177)
    #
    # xgb_rf_classif = xgb.XGBRFClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
    #                          n_estimators=700, objective='binary:logistic', booster='gbtree', n_jobs=4,
    #                          learning_rate=0.2, max_depth=5)
    # #start_time = time.time()
    # xgbrf_model_fit(xgb_rf_classif, X_train, y_train)
    # print("--- %s XGBRF2 seconds ---" % (time.time() - start_time))
    #
    # print("Test  Accuracy Score : %.2f"%xgb_rf_classif.score(X_test, y_test))
    # print("Train Accuracy Score : %.2f"%xgb_rf_classif.score(X_train, y_train))
    #
    # print("Default Number of Estimators : ",xgb_rf_classif.n_estimators)
    # print("Default Max Depth of Trees   : ", xgb_rf_classif.max_depth)
    #
    #
    #
    # #XGBRFClassifier 3
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=177)
    #
    # xgb_rf_classif = xgb.XGBRFClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
    #                          n_estimators=500, objective='binary:logistic', booster='gbtree', n_jobs=4,
    #                          learning_rate=0.5)
    # start_time = time.time()
    # xgbrf_model_fit(xgb_rf_classif, X_train, y_train)
    # print("--- %s XGBRF3 seconds ---" % (time.time() - start_time))
    #
    # print("Test  Accuracy Score : %.2f"%xgb_rf_classif.score(X_test, y_test))
    # print("Train Accuracy Score : %.2f"%xgb_rf_classif.score(X_train, y_train))
    #
    # print("Default Number of Estimators : ",xgb_rf_classif.n_estimators)
    # print("Default Max Depth of Trees   : ", xgb_rf_classif.max_depth)




    #start_time = time.time()





    # params = {
    #     'n_estimators': [300, 500, 700, 1000, 1500, 2000, 5000],
    #     'max_depth': [None, 4, 5, 6],
    #     'eta': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    # }
    # grid_search = GridSearchCV(xgb.XGBRFClassifier(objective='binary:logistic', nthread=4,
    #                                                use_label_encoder=False, predictor='gpu_predictor',
    #                                                tree_method='gpu_hist'), params, n_jobs=-1, cv=5)
    #
    # grid_search.fit(X_train, y_train)
    #
    # print("Test  Accuracy Score : %.2f"%grid_search.score(X_test, y_test))
    # print("Train Accuracy Score : %.2f"%grid_search.score(X_train, y_train))
    #
    # print("Best Params : ", grid_search.best_params_)
    # print("Feature Importances : ")
    # print(pd.DataFrame([grid_search.best_estimator_.feature_importances_], columns=X_test.columns).to_string())
    #print("--- %s XGBRF seconds ---" % (time.time() - start_time))






if __name__ == '__main__':
    main()
