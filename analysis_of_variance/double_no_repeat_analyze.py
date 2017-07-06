from scipy.stats import f
import numpy as np


def solve(data):
    ############################################
    #         calculate sum of squares         #
    ############################################
    a_n = len(data)
    b_n = len(data[0])
    sum_row = []
    sum_column = []
    x_ij_square = 0
    x_ij = 0
    for row in data:
        sum_row.append(sum(row))
    for i in range(b_n):
        temp = 0
        for j in range(a_n):
            temp += data[j][i]
            x_ij_square += data[j][i] ** 2
            x_ij += data[j][i]
        sum_column.append(temp)

    c = x_ij ** 2 / (a_n * b_n)
    s_t = x_ij_square - c
    s_a = sum(map(lambda x: x ** 2, sum_row)) / b_n - c
    s_b = sum(map(lambda x: x ** 2, sum_column)) / a_n - c
    s_e = s_t - s_a - s_b

    ############################################
    #           calculate freedom              #
    ############################################
    freedom_t = a_n * b_n - 1
    freedom_a = a_n - 1
    freedom_b = b_n - 1
    freedom_e = freedom_t - freedom_a - freedom_b

    ############################################
    #      calculate chi-square statistic      #
    ############################################
    v_a = s_a / freedom_a
    v_b = s_b / freedom_b
    v_e = s_e / freedom_e

    ############################################
    #         calculate test statistics        #
    ############################################
    f_a = v_a / v_e
    f_b = v_b / v_e

    f_value = [f_a, f_b]
    p_value = [f.pdf(f_a, freedom_a, freedom_e), f.pdf(f_b, freedom_b, freedom_e)]
    return [f_value, p_value]


# it means use 4 different methods A and 6 different methods B
# row_a means [result(A1, B1), result(A1, B2)...]
row_a = [0.05, 0.46, 0.12, 0.16, 0.84, 1.30]
row_b = [0.08, 0.38, 0.40, 0.10, 0.92, 1.57]
row_c = [0.11, 0.43, 0.05, 0.10, 0.94, 1.10]
row_d = [0.11, 0.44, 0.08, 0.03, 0.93, 1.15]

print(solve((row_a, row_b, row_c, row_d)))
