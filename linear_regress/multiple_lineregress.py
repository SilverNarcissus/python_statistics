from functools import reduce

from numpy import *
from scipy.stats import f, t


def liner_regeress_analyze(array_x, array_y, alpha=0.05):
    ############################################
    #         calculate basic parameter        #
    ############################################
    x = []
    n = len(array_x)
    m = len(array_x[0]) + 1

    for row in array_x:
        x.append([1] + row)
    x = mat(x)
    y = mat(array_y).T
    beta = (x.T * x).I * x.T * y

    h = x * (x.T * x).I * x.T
    i = mat(eye(h.shape[0], h.shape[1], dtype=int))
    error = (i - h) * y

    ############################################
    #         significance analysis            #
    ############################################
    mean_y = mean(array_y)

    s_t = reduce(lambda x, y: x + y, map(lambda y: (y - mean_y) ** 2, array_y))
    s_e = y.T * y - beta.T * x.T * y
    s_r = s_t - s_e

    freedom_regress = m - 1
    freedom_else = n - m
    freedom_total = freedom_regress + freedom_else
    v_r = s_r / freedom_regress
    v_e = s_e / freedom_else
    F = v_r / v_e

    residual_std_error = math.sqrt(v_e)
    interval = -t.ppf(alpha / 2, freedom_else) * residual_std_error

    r = math.sqrt(s_r / s_t)
    significance_very_high = F > f.ppf(0.05, freedom_regress, freedom_else)
    significance_high = F > f.ppf(0.01, freedom_regress, freedom_else)

    ############################################
    # analyze whether each x is significant    #
    ############################################
    variance = (error.T * error).max() / (n - m - 1)
    s_beta = (x.T * x).I
    # noinspection PyTypeChecker
    s_beta = array(variance * s_beta)
    significance_array = []
    significance_conclusion = []
    t_value = -t.ppf(0.025, n - m - 1)
    beta_array = array(beta)
    for i in range(1, m):
        value = float(beta_array[i]) / math.sqrt(s_beta[i][i])
        significance_array.append(value)
        significance_conclusion.append(value > t_value)

    ############################################
    #             return section               #
    ############################################
    if significance_very_high:
        return beta, r, "very high", interval, significance_conclusion
    if significance_high:
        return beta, r, "high", interval, significance_conclusion
    return beta, r, "no", interval, significance_conclusion

# array_x means [[x1(row1), x2(row1)], [x1(row2), x2(row2)]...]
# array_y means [y(row1), y(row2)...]
array_x = [[274, 2450], [180, 3250], [375, 3802], [205, 2838], [86, 2347], [265, 3782], [98, 3008], [330, 2450],
           [195, 2137], [53, 2560], [430, 4020], [372, 4427], [236, 2660], [157, 2088], [370, 2605]]
array_y = [162, 120, 223, 131, 67, 169, 81, 192, 116, 55, 252, 232, 144, 103, 212]
print(liner_regeress_analyze(array_x, array_y))
