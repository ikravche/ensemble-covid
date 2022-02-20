import pandas as pd
import os
import sys
import glob
import argparse
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn import model_selection, metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV  # Perforing grid search


def read_datasets(dataset_path: str):
    all_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    cases_data = pd.concat((pd.read_csv(f, low_memory=False) for f in all_files))
    return cases_data


def preprocess_dataset(tested_individuals_dataset: pd.DataFrame, drop_duplicates: bool):
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
    if drop_duplicates:
        cases_data = cases_data.drop_duplicates()
    return cases_data


def update_dataset_structure(dataset: pd.DataFrame):
    cases_data_copy = dataset.copy()
    corona_results = cases_data_copy.pop('corona_result')
    cases_data_copy = cases_data_copy.drop(['test_date'], axis=1).astype(int)
    cases_data_copy.insert(len(cases_data_copy.columns), corona_results.name, corona_results)
    print(cases_data_copy.iloc[1])
    return cases_data_copy


def gbm_model_fit(alg, x_train, y_train, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
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


def gbm_grid_search_cv(X_train, y_train):
    param_test = {'max_depth': range(5, 16, 2), 'min_samples_split': range(800, 2000, 200)}
    gsearch2 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8,
                                             random_state=10),
        param_grid=param_test, scoring='roc_auc', n_jobs=4, cv=5)
    gsearch2.fit(X_train, y_train)
    print('results')
    print(gsearch2.cv_results_)
    print('params')
    print(gsearch2.best_params_)
    print('scores')
    print(gsearch2.best_score_)


def xgb_grid_search_cv(X_train, y_train):
    param_test1 = {
        'n_estimators': [40, 140, 500, 1000, 2700, 4800, 8400, 10000],
        'eta': [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
    }

    gsearch1 = GridSearchCV(estimator=XGBClassifier(eta=0.01, n_estimators=100, max_depth=6,
                                                    objective='binary:logistic', nthread=6,
                                                    use_label_encoder=False, predictor='gpu_predictor',
                                                    tree_method='gpu_hist'),
                            param_grid=param_test1, scoring='roc_auc', n_jobs=6, cv=5)

    gsearch1.fit(X_train, y_train)

    print('results')
    print(gsearch1.cv_results_)
    print('params')
    print(gsearch1.best_params_)
    print('scores')
    print(gsearch1.best_score_)


def xgbrf_grid_search_cv(X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': [300, 500, 700, 1000, 1500, 2000, 5000],
        'max_depth': [None, 4, 5, 6],
        'eta': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    start_time = time.time()
    grid_search = GridSearchCV(xgb.XGBRFClassifier(objective='binary:logistic', nthread=4,
                                                   use_label_encoder=False, predictor='gpu_predictor',
                                                   tree_method='gpu_hist'), params, n_jobs=-1, cv=5)

    grid_search.fit(X_train, y_train)

    print("Test  Accuracy Score : %.2f" % grid_search.score(X_test, y_test))
    print("Train Accuracy Score : %.2f" % grid_search.score(X_train, y_train))

    print("Best Params : ", grid_search.best_params_)
    print("Feature Importances : ")
    print(pd.DataFrame([grid_search.best_estimator_.feature_importances_], columns=X_test.columns).to_string())
    print("--- %s XGBRF seconds ---" % (time.time() - start_time))


def xgb_model_fit(alg, x_train, y_train):
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


def neural_network(X_train, X_test, Y, y_test):
    X = np.array(X_train)
    x_test = np.array(X_test)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    model = Sequential()
    model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))  # Add an input shape! (features,)
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    start_time = time.time()

    # compile the model
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # early stopping callback
    # This callback will stop the training when there is no improvement in
    # the validation loss for 10 consecutive epochs.
    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',  # don't minimize the accuracy!
                       patience=10,
                       restore_best_weights=True)

    # now we just update our model fit call
    history = model.fit(X,
                        Y,
                        callbacks=[es],
                        epochs=80,  # you can set this to a big number!
                        batch_size=10,
                        validation_split=0.2,
                        shuffle=True,
                        verbose=1)

    history_dict = history.history
    # Learning curve(Loss)
    # let's see the training and validation loss by epoch

    # loss
    loss_values = history_dict['loss']  # you can change this
    val_loss_values = history_dict['val_loss']  # you can also change this

    # range of X (no. of epochs)
    epochs = range(1, len(loss_values) + 1)

    # plot
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Learning curve(accuracy)
    # let's see the training and validation accuracy by epoch

    # accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # range of X (no. of epochs)
    epochs = range(1, len(acc) + 1)

    # plot
    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    # orange is for "orange"
    plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # this is the max value - should correspond to
    # the HIGHEST train accuracy
    np.max(val_acc)
    print(np.max(val_acc))

    model.predict(X)  # prob of successes (survival)
    np.round(model.predict(X), 0)  # 1 and 0 (survival or not)

    # so we need to round to a whole number (0 or 1),
    # or the confusion matrix won't work!
    preds = np.round(model.predict(X), 0)

    # confusion matrix
    print(confusion_matrix(Y, preds))  # order matters! (actual, predicted)

    print(classification_report(Y, preds))

    model.predict(x_test)  # prob of successes (survival)
    np.round(model.predict(x_test), 0)  # 1 and 0 (survival or not)

    # so we need to round to a whole number (0 or 1),
    # or the confusion matrix won't work!
    preds = np.round(model.predict(x_test), 0)

    # confusion matrix
    print(confusion_matrix(y_test, preds))  # order matters! (actual, predicted)

    print(classification_report(y_test, preds))

    print("--- %s seconds ---" % (time.time() - start_time))


def main():

    parser = argparse.ArgumentParser(description='Script so useful.')
    parser.add_argument("--duplicates", type=bool, default=False)
    parser.add_argument("--method", type=str, default='xgb')
    args = parser.parse_args()

    drop_duplicates = args.duplicates
    method = args.method
    print('Executing code with ' + method + ' method')
    duplicates_print = 'without duplicates' if drop_duplicates  else 'with duplicates'
    print(duplicates_print)

    pd.set_option('display.max_columns', 11)
    #hyper_parameters_selection = ''
    os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz\\bin'
    dataset_path = 'datasets/'
    # read dataset from multiple files and concatenate into one dataset
    cases_data = read_datasets(dataset_path=dataset_path)
    # process dataset data - remove columns with empty, None, or misleading values
    cases_data = preprocess_dataset(tested_individuals_dataset=cases_data, drop_duplicates=drop_duplicates)
    # print(cases_data)
    cases_data.to_csv(dataset_path + 'preprocessed_dataset/corona_tested_individuals.csv', index=False)

    # removing columns which will not be used during training
    cases_data = update_dataset_structure(dataset=cases_data)

    print(cases_data.head())

    X, y = cases_data.iloc[:, :-1], cases_data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=146)
    start_time = time.time()

    if method == 'gbm':
        gbm = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=1200,
                                         min_samples_leaf=100, max_depth=5,
                                         max_features='sqrt', loss='deviance',
                                         n_estimators=500)
        gbm_model_fit(gbm, X_train, y_train, X_train.columns)
        print("--- %s GBM seconds ---" % (time.time() - start_time))
        predictions = gbm.predict(X_test)
        print("F1 on test set: %.4g" % metrics.f1_score(y_test.values, predictions))
        print("Accuracy on test set: %.4g" % metrics.accuracy_score(y_test.values, predictions))

    elif method == 'xgb':
        xgb = XGBClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
                            n_estimators=400, objective='binary:logistic', booster='gbtree', n_jobs=4, eta=0.3)

        xgb_model_fit(xgb, X_train, y_train)
        print("--- %s XGB seconds ---" % (time.time() - start_time))
        predictions = xgb.predict(X_test)
        print("F-score on test set: %.4g" % metrics.f1_score(y_test.values, predictions))
        print("Accuracy on test set: %.4g" % metrics.accuracy_score(y_test.values, predictions))

    elif method == 'xgbrf':
        xgb_rf_classif = XGBRFClassifier(use_label_encoder=False, predictor='gpu_predictor', tree_method='gpu_hist',
                                         n_estimators=20, objective='binary:logistic', booster='gbtree', n_jobs=4,
                                         learning_rate=0.5, max_depth=4)

        xgbrf_model_fit(xgb_rf_classif, X_train, y_train)
        print("--- %s XGBRF seconds ---" % (time.time() - start_time))

        print("Test  Accuracy Score : %.2f" % xgb_rf_classif.score(X_test, y_test))
        print("Train Accuracy Score : %.2f" % xgb_rf_classif.score(X_train, y_train))

        predictions = xgb_rf_classif.predict(X_test)
        print("F-score XGBRF on test 20  set: %.4g" % metrics.f1_score(y_test.values, predictions))
        print("Accuracy on test set: %.4g" % metrics.accuracy_score(y_test.values, predictions))

    elif method == 'nn':
        neural_network(X_train=X_train, X_test=X_test, y_test=y_test, Y=y)
    else:
        print('Method was not recognized')
        sys.exit(1)


if __name__ == '__main__':
    main()
