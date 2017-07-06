from scipy.stats import f
import numpy as np


def solve(data):
    ############################################
    #         calculate sum of squares         #
    ############################################
    a_n = len(data)
    b_n = len(data[0])
    t_n = len(data[0][0])
    sum_row = []
    sum_column = []
    x_ijk_square = 0
    x_ijk = 0
    x_ij_square = 0
    for row in data:
        sum_row.append(np.sum(row))
    for i in range(b_n):
        temp1 = 0
        for j in range(a_n):
            temp2 = 0
            for k in range(t_n):
                temp2 += data[j][i][k]
                temp1 += data[j][i][k]
                x_ijk_square += data[j][i][k] ** 2
                x_ijk += data[j][i][k]
            x_ij_square += temp2 ** 2
        sum_column.append(temp1)

    c = x_ijk ** 2 / (a_n * b_n * t_n)
    s_t = x_ijk_square - c
    s_a = sum(map(lambda x: x ** 2, sum_row)) / (b_n * t_n) - c
    s_b = sum(map(lambda x: x ** 2, sum_column)) / (a_n * t_n) - c
    s_e = x_ijk_square - x_ij_square / t_n
    s_ab = s_t - s_a - s_b - s_e

    ############################################
    #           calculate freedom              #
    ############################################
    freedom_t = a_n * b_n * t_n - 1
    freedom_a = a_n - 1
    freedom_b = b_n - 1
    freedom_ab = freedom_a * freedom_b
    freedom_e = freedom_t - freedom_a - freedom_b - freedom_ab

    ############################################
    #      calculate chi-square statistic      #
    ############################################
    v_a = s_a / freedom_a
    v_b = s_b / freedom_b
    v_ab = s_ab / freedom_ab
    v_e = s_e / freedom_e

    ############################################
    #         calculate test statistics        #
    ############################################
    f_a = v_a / v_e
    f_b = v_b / v_e
    f_ab = v_ab / v_e

    f_value = [f_a, f_b, f_ab]
    p_value = [f.pdf(f_a, freedom_a, freedom_e), f.pdf(f_b, freedom_b, freedom_e), f.pdf(f_ab, freedom_ab, freedom_e)]
    return [f_value, p_value]


# it means use 4 different methods A and 3 different methods B each combine of method test 2 times
# row_a means [[result1(A1, B1), result2(A1, B1)],[result1(A1, B2), result2(A1, B2)]...]
row_a = [[58.2, 52.6], [56.2, 41.2], [65.3, 60.8]]
row_b = [[49.1, 42.8], [54.1, 50.5], [51.6, 48.4]]
row_c = [[60.1, 58.3], [70.9, 73.2], [39.2, 40.7]]
row_d = [[75.8, 71.5], [58.2, 51.0], [48.7, 41.4]]

print(solve((row_a, row_b, row_c, row_d)))
