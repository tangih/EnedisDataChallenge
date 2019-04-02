import numpy as np
import tqdm

from keras.models import Sequential
from keras.layers import Dense


def train_models(x_train, y_train, hh_train):
    models = [None for i in range(48)]
    seed = 7
    np.random.seed(seed)
    batch_size = 32
    n_epochs = 100
    n_out = y_train.shape[1]
    for i in tqdm.tqdm_notebook(range(48)):
        models[i] = Sequential()
        models[i].add(Dense(32, input_dim=30, kernel_initializer='normal', activation='relu'))
        models[i].add(Dense(32, input_dim=32, kernel_initializer='normal', activation='relu'))
        models[i].add(Dense(16, input_dim=32, kernel_initializer='normal', activation='relu'))
        models[i].add(Dense(n_out, input_dim=16, kernel_initializer='normal'))
        models[i].compile(loss='mean_squared_error', optimizer='adam')
        ind = np.where(hh_train == i)[0]
        x_train_i, y_train_i = x_train[ind], y_train[ind]
        ind = np.arange(x_train_i.shape[0])
        np.random.shuffle(ind)
        x_train_i = x_train_i[ind]
        y_train_i = y_train_i[ind]
        # print('Training model {:2d}/{} with {} samples'.format(i+1, 48, len(ind)))
        models[i].fit(x_train_i, y_train_i, epochs=n_epochs, batch_size=batch_size, verbose=0)
    return models

def predict_multi(models, X_test, hh_test):
    _, n_out = models[0].get_layer(index=-1).output.shape
    Y_test = np.zeros((X_test.shape[0], n_out))
    inds = []
    for i in tqdm.tqdm_notebook(range(48)):
        ind = np.where(hh_test == i)[0]
        # print('Testing model {:2d}/{} on {} samples'.format(i+1, 48, ind.shape[0]))
        X_test_hh = X_test[ind]
        Y_test_hh = models[i].predict(X_test_hh)
        Y_test[ind] = Y_test_hh
    return Y_test