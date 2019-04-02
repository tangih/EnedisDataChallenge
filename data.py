import datetime
import time
import numpy as np

def holiday_list():
    holidays = []
    with open('joursferies.txt', 'r') as f:
        for line in f:
            holidays.append(line[:-1])
    return holidays


def indiv_dayclass(date, holidays):
    datestring = '{:02d}/{:02d}/{}'.format(date.day, date.month, date.year)
    if date.weekday() == 5 or date.weekday() == 6 or datestring in holidays:
        return 'H'
    return 'W'


def dayclass(t, holidays, min_timestamp=1381694400):
    today = datetime.datetime.fromtimestamp(t + min_timestamp)
    yesterday = today - datetime.timedelta(1)
    return indiv_dayclass(yesterday, holidays) + indiv_dayclass(today, holidays)
    

def separate_x_res(N):
    is_valid = [False for i in range(N)]
    for m in range(12):
        if m%2 == 0:
            continue
        for i in range(m*48*30, (m+1)*48*30):
            is_valid[i] = True
    return is_valid


def separate_x_com(n_com):
    # k_sep = np.argmax([x_com[i+1, 1] - x_com[i, 1] for i in range(n_com-1)])
    k_sep = n_com * .9
    is_valid = np.arange(n_com) > k_sep
    return is_valid


def create_features(x_train, min_timestamp=1381694400):
    holidays = holiday_list()
    day_labels = ['HH', 'HW', 'WH', 'WW']
    n, m = x_train.shape
    features = np.zeros((n, 30), dtype=np.float)
    for i in range(n):
        vec = np.zeros(30, dtype=np.float)
        t = x_train[i, 1]
        dt = datetime.datetime.fromtimestamp(t + min_timestamp)
        weekday = dt.weekday()
        dc = dayclass(t, holidays)
        vec[day_labels.index(dc)] = 1  # dayclass (one hot)
        vec[4+weekday] = 1  # day of the week (one hot)
        beg_year = time.mktime(datetime.datetime.strptime('{}'.format(dt.year), '%Y').timetuple())
        timestamp = t + min_timestamp - beg_year
        vec[11] = timestamp  # time from the beginning of the year
        vec[12:25] = x_train[i][3:16]
        vec[25:28] = x_train[i][18:21]
        vec[28:30] = x_train[i][23:25]
        features[i] = vec
    return features


def create_sets(features, x, y, is_valid, min_timestamp=1381694400):
    n_feats = features.shape[0]
    halfhour = np.zeros(n_feats, dtype=np.int)
    for i in range(n_feats):
        t = x[i, 1]
        date = datetime.datetime.fromtimestamp(t+min_timestamp)
        h, m = date.hour, date.minute
        n = h * 2 + (m // 30)
        halfhour[i] = n

    id_valid = np.where(is_valid)[0]
    x_valid = features[id_valid]
    y_valid = y[id_valid]
    hh_valid = halfhour[id_valid]
    id_train = np.where([not is_valid[i] for i in range(x.shape[0])])[0]
    x_train = features[id_train]
    y_train = y[id_train]
    hh_train = halfhour[id_train]
    
    return x_train, y_train, hh_train, x_valid, y_valid, hh_valid, id_train, id_valid
