from scipy import stats, math
import numpy as np
import pylab
from scipy.stats import f, t


def liner_regeress_analyze(array_x, array_y, alpha=0.05):
    ############################################
    #         calculate basic parameter        #
    ############################################
    x = np.array(array_x)
    y = np.array(array_y)
    mean_y = np.mean(y)
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, y)
    predict_y = intercept + slope * x

    ############################################
    #         significance analysis            #
    ############################################
    s_t = sum(map(lambda y: (y - mean_y) ** 2, array_y))
    s_r = sum(map(lambda y: (y - mean_y) ** 2, predict_y))
    s_e = sum(map(lambda y1, y2: (y1 - y2) ** 2, array_y, predict_y))

    freedom_regress = 1
    freedom_else = len(array_x) - 2
    freedom_total = freedom_regress + freedom_else

    v_r = s_r / freedom_regress
    v_e = s_e / freedom_else

    # s_y
    residual_std_error = math.sqrt(v_e)
    interval = -t.ppf(alpha / 2, freedom_else) * residual_std_error
    F = v_r / v_e

    significance_very_high = F > f.ppf(0.05, freedom_regress, freedom_else)
    significance_high = F > f.ppf(0.01, freedom_regress, freedom_else)

    ############################################
    #             return section               #
    ############################################
    if significance_very_high:
        return slope, intercept, r_value, p_value, "very high", interval
    if significance_high:
        return slope, intercept, r_value, p_value, "high", interval
    return slope, intercept, r_value, p_value, "no", interval


# x means [x(row1), x(row2)...]
# y means [y(row1), y(row2)...]
x = [255.7, 263.3, 275.4, 278.3, 296.7, 309.4, 315.8, 318.8, 330.0, 340.2, 350.7, 367.3, 381.3, 406.5, 430.8, 451.5]
y = [116.5, 120.8, 124.4, 125.5, 131.7, 136.2, 138.7, 140.2, 146.8, 149.6, 153.0, 158.2, 163.2, 170.5, 178.2, 185.9]
print(liner_regeress_analyze(x, y))
#
# predict_y = intercept + slope * x
# pred_error = y - predict_y
# degrees_of_freedom = len(x) - 2
# residual_std_error = np.sqrt(np.sum(pred_error ** 2) / degrees_of_freedom)
#
# # Plotting
# # pylab.plot(x, y, 'o')
# # pylab.plot(x, predict_y, 'k-')
# # pylab.show()
#
# print((slope, intercept, r_value, p_value, slope_std_error, residual_std_error))
