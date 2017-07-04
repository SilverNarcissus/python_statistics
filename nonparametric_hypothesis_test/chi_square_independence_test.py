import math
from functools import reduce
from unicodedata import decimal
from scipy.stats import chi2


def solve(data):
    ##################################
    #        data pre-process        #
    ##################################
    alpha = 0.01
    row_total = []
    column_total = []
    row_count = len(data)
    column_count = len(data[0])
    for i in range(column_count):
        column_total.append(0)
    for row in data:
        row_total.append(reduce(lambda x, y: x + y, row))
        for i in range(column_count):
            column_total[i] += row[i]
    total = reduce(lambda x, y: x + y, row_total)

    ##################################
    #          data analyze          #
    ##################################
    freedom = (row_count - 1) * (column_count - 1)
    ka = 0
    for i in range(row_count):
        for j in range(column_count):
            temp = (row_total[i] * column_total[j] / total)
            ka += (data[i][j] - temp) ** 2 / temp

    conclusion = ka < chi2.ppf(1 - alpha, freedom)

    return [freedom, round(ka, 2), conclusion]
    # the test result is 11.47


a = [186, 38, 35]
b = [227, 54, 45]
c = [219, 78, 78]
d = [355, 112, 140]
e = [653, 285, 259]
data = [a, b, c, d, e]
print(solve(data))
print(chi2.ppf(1 - 0.01, 8))
