import numpy as np
import pandas as pd
import random
import sys

#np.set_printoptions(threshold=sys.maxsize)

pd.set_option('display.max_columns', None)


def get_data(s, p: int | None = None):
    data_raw = pd.read_csv('/home/admin/Uni/ub_code/wheatX.csv', sep=";")

    # get head of table to seperate x and y
    # there are 2 possible start strings for each item in head "c" and "wPt"
    head = [item[0] for item in data_raw.items()]
    values = np.transpose(data_raw.values)
    x_beta, y = [], []
    for i, name in enumerate(head):  # sort input data into x and y values
        if str(name)[0] == "c":
            x_beta.append(values[i])
        elif str(name)[0] == "w":
            y.append(values[i])

    x_beta = np.array(x_beta).transpose()
    y = np.array(y).transpose()  # transfrom to np.array

    if p is not None:  # check if some columns should be filtered out
        rand_indexes = np.random.choice(x_beta.shape[1], p, replace=False)
        x_beta = x_beta[:, rand_indexes]

    # calculate epsilon based on shape and standart deviation
    sig = np.var(x_beta, ddof=1)  # ddof=1 for empirical variance
    eps = np.random.normal(0, s * sig, size=(x_beta.shape[0], 1))
    return y, x_beta, eps


def get_test_data(alpha: float):
    if alpha > 1:
        raise ValueError("Alpha needs to be below one")

    y, x_beta, eps = get_data(1)
    num_individuals = (int)(y.shape[1] * alpha)
    rand_indexes = np.random.choice(x_beta.shape[0],
                                    num_individuals,
                                    replace=False
                                    )
    x_beta = x_beta[:, rand_indexes]
    eps = eps[:, rand_indexes]
    y = y[:, rand_indexes]
    return y, x_beta, eps, rand_indexes


# Question 1
y, x_beta, eps = get_data(s=2, p=10)
print(y.shape, x_beta.shape, eps.shape)

# Question 2
y, x_beta, eps = get_test_data(s=2, p=10)
