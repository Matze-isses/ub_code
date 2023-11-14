import os import name
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
    x, y = [], []
    for i, name in enumerate(head):  # sort input data into x and y values
        if str(name)[0] == "c":
            x.append(values[i])
        elif str(name)[0] == "w":
            y.append(values[i])

    x = np.array(x).transpose()
    y = np.array(y).transpose()  # transfrom to np.array

    if p is not None:  # check if some columns should be filtered out
        rand_indexes = np.random.choice(x.shape[1], p, replace=False)
        x = x[:, rand_indexes]

    # calculate epsilon based on shape and standart deviation
    sig = np.var(x, ddof=1)  # ddof=1 for empirical variance
    eps = np.random.normal(0, s * sig, size=(x.shape[0], 1))
    return y, x, eps


def get_test_data(alpha: float):
    if alpha > 1:
        raise ValueError("Alpha needs to be below one")

    y, x, eps = get_data(1)
    num_individuals = (int)(y.shape[1] * alpha)
    rand_indexes = np.random.choice(x.shape[0],
                                    num_individuals,
                                    replace=False
                                    )
    y_test = y[rand_indexes, :]
    x_test = x[rand_indexes, :]
    eps_test = eps[rand_indexes, :]
    return y, x, eps, y_test, x_test, eps_test, rand_indexes


# Question 1
y, x, eps = get_data(s=2, p=10)
print(y.shape, x.shape, eps.shape)

# Question 2
result = get_test_data(alpha=0.2)
names = ["y", "x", "eps"]
print(" Question 2 ")
print(" " * 10 + "| Train Shape " + "| Test Shape")
print("-" * 40)
for i in range(3):
    shape_train = str(result[i].shape)
    shape_test = str(result[3 + i].shape)
    print(names[i] + " " * (10 - len(names[i])) + "| "
          + shape_train + " " * (11 - len(shape_train))
          + " | " + shape_test)
print("")
print("The test individuals are: ", result[-1])
