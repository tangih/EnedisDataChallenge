import datetime
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