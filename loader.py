import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import tqdm

def convert(str):
    if str == '':
        return 0
    return float(str)


def load_data(train_file_feat, train_file_label, test_file_feat, min_timestamp = 1381694400):
    # load raw data
    x_train = []
    y_train = []
    x_test = []
    with open(train_file_feat, 'r') as f:
        for line in f:
            x_train.append(line[:-1])
    with open(test_file_feat, 'r') as f:
        for line in f:
            x_test.append(line[:-1])
    with open(train_file_label, 'r') as f:
        for line in f:
            y_train.append(line[:-1])
    col_names = x_train[0].split(',')
    x_train = [line.split(',') for line in x_train[1:]]
    label_names = y_train[0].split(',')
    y_train = [line.split(',') for line in y_train[1:]]

    # convert data into numpy array
    N = len(x_train)
    data = np.zeros((N, 25))
    labels = ['IDS', 'Timestamp']
    for j in range(3, len(col_names)):
        labels.append(col_names[j])
    
    for i in tqdm.tqdm_notebook(range(N), 'Loading data:'):
        row = []
        assert(len(x_train[i]) == 26)
        for j in range(26):
            if j == 1:
                # convert UTC time into translated timestamp
                timestamp = time.mktime(datetime.datetime.strptime(x_train[i][1], '%d/%m/%Y %H:%M').timetuple())
                row.append(timestamp - min_timestamp)
            elif j == 0 or j >= 3:
                row.append(convert(x_train[i][j]))
        data[i] = np.array(row)
       
    # separate commercial and residential data
    min_i = 0
    for i in range(N):
        if y_train[i][-1] != 'NA':
            min_i = i
            break
    x_com = data[min_i:]
    x_res = data
    y_com = np.zeros((N - min_i, 3), dtype=np.float)
    y_res = np.zeros((N, 4), dtype=np.float)
    res_ind = [4, 5, 7, 8]
    com_ind = [6, 9, 10]
    for i in range(data.shape[0]):
        y_res[i] = np.array([convert(x) for x in [y_train[i][j] for j in res_ind]])
        if i >= min_i:
            y_com[i-min_i] = np.array([convert(x) for x in [y_train[i][j] for j in com_ind]])
    return x_res, y_res, x_com, y_com, labels
