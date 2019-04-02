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
    
    
def plot_time(t, X, label_x=None, label_curve=None, alpha=1):
    if label_curve is not None:
        plt.plot(t, X, label=label_curve, alpha=alpha)
    else:
        plt.plot(t, X, alpha=alpha)
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
    

def add(X, x):
    if X.shape[0] == 0:
        return x
    else:
        return np.concatenate((X, x), axis=0)    
    

def plot_dayclass(x_train, y_train, x_labels, x_id, y_id, min_id, max_id, min_timestamp=1381694400):
    X = [np.array([]), np.array([]), np.array([]), np.array([])]  # HH, HW, WH, WW
    Y = [np.array([]), np.array([]), np.array([]), np.array([])]

    for k in range(7):
        x, y = x_train[k*48+min_id:max_id:48*7], y_train[k*48+min_id:max_id:48*7]
        t = x_train[k*48+min_id, 1]
        day = datetime.datetime.fromtimestamp(t + min_timestamp).weekday()
        day_labels = ['HH', 'HW', 'WH', 'WW']
        if day == 0:
            id = day_labels.index('HW')
            X[id] = add(X[id], x)
            Y[id] = add(Y[id], y)
        elif day == 1 or day == 2 or day ==3 or day == 4:
            id = day_labels.index('WW')
            X[id] = add(X[id], x)
            Y[id] = add(Y[id], y)
        elif day == 5:
            id = day_labels.index('WH')
            X[id] = add(X[id], x)
            Y[id] = add(Y[id], y)
        elif day == 6:
            id = day_labels.index('HH')
            X[id] = add(X[id], x)
            Y[id] = add(Y[id], y)

    for k in range(4):
        ind = np.argsort(X[k][:, 0])
        X[k] = X[k][ind]
        Y[k] = Y[k][ind]

    # vis.plot_all(X, Y, x_labels, filename='plots_by_class.png')

    for k in range(4):
        plot_series(X[k][:, x_id], Y[k][:, 0], label_a=x_labels[x_id], label_b='YRES{}'.format(y_id), label=day_labels[k])
    plt.legend(loc='best')
    
    
def plot_train_val(x, y, y_id, is_valid, x_labels):
    n = x.shape[0]
    print('Proportion of the validation set: {:.1f}%'.format(100 * np.sum(is_valid)/n))
    # plot training/validation set
    ind = np.where(is_valid)[0]
    plot_time(x[ind, 1], y[ind, 0], label_curve='validation')
    ind = np.where([not is_valid[i] for i in range(n)])[0]
    plot_time(x[ind, 1], y[ind, 0], label_curve='training')
    plt.legend(loc='best')

def score(y_pred, y_gt, plot=False, x=None, id_valid=None):
    se = (y_pred - y_gt) ** 2
    mean_se = np.mean(se, axis=0)
    n_out = y_pred.shape[1]
    for i in range(n_out):
        print('SE for Y{}: {:.4f}'.format(i, mean_se[i]))
    print('MSE RES: {:.5f}'.format(np.mean(mean_se)))
    print('Mean relative error: {:.2f}%'.format(np.mean(mean_se / np.mean(y_gt, axis=0)) * 100))

    if plot:
        plt.figure(figsize=(15, 5))
        for i in range(n_out):
            t = x[id_valid, 1]
            plot_time(t[:500], np.log(se[:500, i] / 1e-5), label_x='log MSE', label_curve=str(i))
        plt.legend(loc='best')
        plt.show()
