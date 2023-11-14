from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats

show_solution = 2


def generate_data(function: Callable, interval=(0, 30)) -> Tuple[float, float]:
    x = np.arange(interval[0], interval[1])
    y = function(x)

    y = y.tolist()
    if type(y[0]) is list:
        y = y[0]

    return x.tolist(), y


def do_one_iteration(return_beta: bool = False, use_alt_function: bool = False):
    if use_alt_function:
        function = lambda x: (5 + 3 * x + scipy.stats.cauchy.rvs(loc=0, scale=7, size=30))  # defined on worksheet (4)
    else:
        function = lambda x: (5 + 3 * x + np.random.normal(0, 49, (1, 30)))  # defined on worksheet (1)

    real_values = lambda x: 5 + 3 * x  # generate function to create the real curve

    x, y = generate_data(function)
    x_real, y_real = generate_data(real_values)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # You can access the estimated parameters directly
    b_0 = model.params[0]
    b_1 = model.params[1]

    if return_beta:
        return b_0, b_1, 0, 0, 0, 0

    estimated_beta = lambda x: (b_0 + b_1 * x)  # generate the function for the estimated regression curve

    x_estimate, y_estimate = generate_data(estimated_beta)
    return (x, y, x_real, y_real, x_estimate, y_estimate)


fig = plt.figure()
ax = fig.add_subplot()

if show_solution == 1:  # plot estimation of beta
    x, y, x_real, y_real, x_estimate, y_estimate = do_one_iteration(return_beta=False)
    ax.plot(x, y, c='g')
    ax.plot(x_real, y_real, c='b')

    ax.plot(x_estimate, y_estimate, c='r')

elif show_solution == 2:  # plot beta for 1000 iterations
    b = [[], []]
    for _ in range(10000):
        b_0, b_1, _, _, _, _ = do_one_iteration(return_beta=True)
        b[0].append(b_0)
        b[1].append(b_1)

    ax.scatter(b[0], b[1], s=0.2)

elif show_solution == 3:  # use cauchy
    b = [[], []]
    for _ in range(1000):
        b_0, b_1, _, _, _, _ = do_one_iteration(return_beta=True, use_alt_function=True)
        b[0].append(b_0)
        b[1].append(b_1)
    ax.scatter(b[0], b[1])

if show_solution > 1:  # if b_0, b_1 are displayed than plot real values
    ax.scatter([5], [3], c='r')

plt.show()



