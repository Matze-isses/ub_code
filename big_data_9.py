import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats

#np.set_printoptions(threshold=sys.maxsize)

pd.set_option('display.max_columns', None)


def get_data():
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
    return x, y


def filter_data(s, p: int, X, Y):
    print(p, X.shape)
    rand_indexes = np.random.choice(X.shape[0], p, replace=False)
    y = Y[rand_indexes, :]
    x = X[rand_indexes, :]

    # calculate epsilon based on shape and standart deviation
    sig = np.var(X, ddof=1)  # ddof=1 for empirical variance
    eps = np.random.normal(0, s * sig, size=(X.shape[0], 1))
    return y, x, eps


def get_test_data(alpha: float):
    if alpha > 1:
        raise ValueError("Alpha needs to be below one")

    x, y = get_data()
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
X, Y = get_data()
y, x, eps = filter_data(1, 100, X,Y)
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
print("")
print("Indexes which are used as test data")
sublist_print = [result[-1][i*5:(i + 1) * 5]
                 for i in range((int)(len(result[-1]) / 5))]
for item in sublist_print:
    print('{:3d} {:3d} {:3d} {:3d} {:3d}'.format(
        item[0], item[1], item[2], item[3], item[4]))

# Question 3
x, y = get_data()
x_p, y_p, _ = filter_data(1, 200, x, y)
x_q, y_q, _ = filter_data(1, 100, x_p, y_p)
model = sm.OLS(y_q, x_q).fit()
print(model)


