# modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from datetime import datetime
import time


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
    # cases_data = cases_data.drop_duplicates()
    return cases_data


def update_dataset_structure(dataset: pd.DataFrame):
    cases_data_copy = dataset.copy()
    corona_results = cases_data_copy.pop('corona_result')
    cases_data_copy = cases_data_copy.drop(['test_date'], axis=1).astype(int)
    cases_data_copy.insert(len(cases_data_copy.columns), corona_results.name, corona_results)
    print(cases_data_copy.iloc[1])
    return cases_data_copy


def main():
    dataset_path = 'datasets/'
    # read dataset from multiple files and concatenate into one dataset
    cases_data = read_datasets(dataset_path=dataset_path)
    # process dataset data - remove columns with empty, None, or misleading values
    cases_data = preprocess_dataset(tested_individuals_dataset=cases_data)
    # print(cases_data)
    cases_data.to_csv(dataset_path + 'preprocessed_dataset/corona_tested_individuals.csv', index=False)

    # removing columns which will not be used during training
    cases_data = update_dataset_structure(dataset=cases_data)

    X, y_init = cases_data.iloc[:, :-1], cases_data.iloc[:, -1]

    X_train, X_test, Y, y_test = train_test_split(X, y_init, test_size=0.2, random_state=146)

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


if __name__ == '__main__':
    main()
