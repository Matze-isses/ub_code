import numpy as np
import pandas as pd
import pprint
import os

from pandas.io.formats.printing import pprint_thing

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


def get_data(sig, n):
    data_raw = pd.read_csv('/home/admin/Uni/ub_code/wheatX.csv')
    data = {}
    values = []
    for x in data_raw:
        values = str(x).split(".")
        values = [value_.replace("\"", "").split(";") for value_ in values]

    for item in values:
        if len(item) <= 1:
            continue
        name = item[1]
        value = item[0]
        if name not in data:
            data[name] = []
        data[name].append((float)(value) if value != '' else np.nan)

    pprint.pprint(data)

#   data = np.array(data)
#   mean_x = np.mean(data)
#   sig = np.sqrt(1 / (len(X) - 1) * sum([(x - mean_x) ** 2 for x in X]))
#   print(sig)
#   eps = np.random.normal(0, sig * np.identity(n))


get_data(1,100)
