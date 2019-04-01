import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


def timestamp2utc(t, min_timestamp=1381694400):
    timestamp = t + min_timestamp
    return datetime.datetime.fromtimestamp(timestamp).isoformat()


def plot_series(A, B, label_a=None, label_b=None, label=None):
    ind = np.argsort(A)
    x = A[ind]
    y = B[ind]
    if label_a is not None and label_b is not None:
        plt.xlabel(label_a)
        plt.ylabel(label_b)
    if label is not None:
        plt.scatter(x, y, marker='+', label=label)
    else:
        plt.scatter(x, y, marker='+')
    
def plot_time(t, X, label_x=None, label_curve=None):
    if label_curve is not None:
        plt.plot(t, X, label=label_curve)
    else:
        plt.plot(t, X)
    min_t, max_t = min(t), max(t)
    if label_x is not None:
        plt.ylabel(label_x)
        plt.title('Temporal plot of {} between {} and {}'.format(label_x, 
                                                                 timestamp2utc(min_t), 
                                                                 timestamp2utc(max_t)))
    else:
        plt.title('Temporal plot between {} and {}'.format(timestamp2utc(min_t), 
                                                           timestamp2utc(max_t)))
        
def plot_all(X, Y, x_labels, filename='plots.png'):
    columns = 22
    rows = Y[0].shape[1]
    n_plots = len(X)
    fig, ax_array = plt.subplots(rows, columns,squeeze=False, figsize=(125, 22))
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            x_id = j + 3
            y_id = i
            for k in range(n_plots):
                x = X[k][:, x_id]
                y = Y[k][:, y_id]
                axes.scatter(x, y, marker='+')

            label_x = x_labels[x_id]
            label_y = label_b='YRES{}'.format(y_id)
            axes.set_xlabel(label_x)
            axes.set_ylabel(label_y)
    fig.savefig(filename)
    
X = [np.array([]), np.array([]), np.array([]), np.array([])]  # HH, HW, WH, WW
Y = [np.array([]), np.array([]), np.array([]), np.array([])]

def add(X, x):
    if X.shape[0] == 0:
        return x
    else:
        return np.concatenate((X, x), axis=0)

def plot_dayclass(x_train, y_train, x_labels, x_id, y_id, min_id, max_id, min_timestamp=1381694400):
    
    for k in range(7):
        x, y = x_train[k*48+min_id:max_id:48*7], y_train[k*48+min_id:max_id:48*7]
        t = x_train[k*48+min_id, 1]
        day = datetime.datetime.fromtimestamp(t + min_timestamp).weekday()
        if day == 0:
            X[1] = add(X[1], x)
            Y[1] = add(Y[1], y)
        elif day == 1 or day == 2 or day ==3 or day == 4:
            X[3] = add(X[3], x)
            Y[3] = add(Y[3], y)
        elif day == 5:
            X[2] = add(X[2], x)
            Y[2] = add(Y[2], y)
        elif day == 6:
            X[0] = add(X[0], x)
            Y[0] = add(Y[0], y)

    for k in range(4):
        ind = np.argsort(X[k][:, 0])
        X[k] = X[k][ind]
        Y[k] = Y[k][ind]

    # vis.plot_all(X, Y, x_labels, filename='plots_by_class.png')

    day_labels = ['HH', 'HW', 'WH', 'WW']
    for k in range(4):
        plot_series(X[k][:, x_id], Y[k][:, 0], label_a=x_labels[x_id], label_b='YRES{}'.format(y_id), label=day_labels[k])
    plt.legend(loc='best')
