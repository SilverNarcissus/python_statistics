from functools import reduce
import numpy as np
from scipy.stats import f, t, stats
from machine_learning.evaluator import regression_evaluator


class LineRegress:
    def __init__(self, beta, r, total_significance, interval, significance_conclusion):
        self.beta = beta
        self.r = r
        self.total_significance = total_significance
        self.interval = interval
        self.significance_conclusion = significance_conclusion

    def predict(self, simple_x):
        if not isinstance(simple_x, list):
            simple_x = [simple_x]
        x = np.mat(np.append([1], simple_x))
        print(x)
        return float(x * self.beta)

    @staticmethod
    def fit(array_x, array_y, alpha=0.05):
        if isinstance(array_x[0], list):
            return LineRegress.fit_multi(array_x, array_y, alpha)

        return LineRegress.fit_single(array_x, array_y, alpha)

    @staticmethod
    def fit_single(array_x, array_y, alpha=0.05):
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

        v_r = s_r/ freedom_regress
        v_e = s_e / freedom_else

        # s_y
        residual_std_error = np.math.sqrt(v_e)
        interval = -t.ppf(alpha / 2, freedom_else) * residual_std_error
        F = v_r / v_e

        significance_very_high = F > f.ppf(0.05, freedom_regress, freedom_else)
        significance_high = F > f.ppf(0.01, freedom_regress, freedom_else)

        ############################################
        #             return section               #
        ############################################
        if significance_very_high:
            return LineRegress(np.mat([intercept, slope]).T, r_value, "very high", interval, None)
        if significance_high:
            return LineRegress(np.mat([intercept, slope]).T, r_value, "high", interval, None)
        return LineRegress(np.mat([intercept, slope]).T, r_value, "no", interval, None)

    @staticmethod
    def fit_multi(array_x, array_y, alpha=0.05):
        ############################################
        #         calculate basic parameter        #
        ############################################
        x = []
        n = len(array_x)
        m = len(array_x[0]) + 1

        for row in array_x:
            x.append([1] + row)
        x = np.mat(x)
        y = np.mat(array_y).T
        beta = (x.T * x).I * x.T * y

        h = x * (x.T * x).I * x.T
        i = np.mat(np.eye(h.shape[0], h.shape[1], dtype=int))
        error = (i - h) * y

        ############################################
        #         significance analysis            #
        ############################################
        mean_y = np.mean(array_y)

        s_t = reduce(lambda x, y: x + y, map(lambda y: (y - mean_y) ** 2, array_y))
        s_e = y.T * y - beta.T * x.T * y
        s_r = s_t - s_e

        freedom_regress = m - 1
        freedom_else = n - m
        freedom_total = freedom_regress + freedom_else
        v_r = s_r / freedom_regress
        v_e = s_e / freedom_else
        F = v_r / v_e

        residual_std_error = np.math.sqrt(abs(v_e))
        interval = -t.ppf(alpha / 2, freedom_else) * residual_std_error

        r = np.math.sqrt(s_r / s_t)
        significance_very_high = F > f.ppf(0.05, freedom_regress, freedom_else)
        significance_high = F > f.ppf(0.01, freedom_regress, freedom_else)

        ############################################
        # analyze whether each x is significant    #
        ############################################
        variance = (error.T * error).max() / (n - m - 1)
        s_beta = (x.T * x).I
        # noinspection PyTypeChecker
        s_beta = np.array(variance * s_beta)
        significance_array = []
        significance_conclusion = []
        t_value = -t.ppf(0.025, n - m - 1)
        beta_array = np.array(beta)
        for i in range(1, m):
            value = float(beta_array[i]) / np.math.sqrt(s_beta[i][i])
            significance_array.append(value)
            significance_conclusion.append(value > t_value)

        ############################################
        #             return section               #
        ############################################
        if significance_very_high:
            return LineRegress(beta, r, "very high", interval, significance_conclusion)
        if significance_high:
            return LineRegress(beta, r, "high", interval, significance_conclusion)
        return LineRegress(beta, r, "no", interval, significance_conclusion)


# array_x means [[x1(row1), x2(row1)], [x1(row2), x2(row2)]...]
# array_y means [y(row1), y(row2)...]
array_x = [[274, 2450], [180, 3250], [375, 3802], [205, 2838], [86, 2347], [265, 3782], [98, 3008], [330, 2450],
           [195, 2137], [53, 2560], [430, 4020], [372, 4427], [236, 2660], [157, 2088], [370, 2605]]
array_y = [162, 120, 223, 131, 67, 169, 81, 192, 116, 55, 252, 232, 144, 103, 212]
x = [255.7, 263.3, 275.4, 278.3, 296.7, 309.4, 315.8, 318.8, 330.0, 340.2, 350.7, 367.3, 381.3, 406.5, 430.8, 451.5]
y = [116.5, 120.8, 124.4, 125.5, 131.7, 136.2, 138.7, 140.2, 146.8, 149.6, 153.0, 158.2, 163.2, 170.5, 178.2, 185.9]
result = LineRegress.fit(array_x[:10], array_y[:10])
print(regression_evaluator(result.predict, np.array(array_x[10:15]), np.array(array_y[10:15])))

result = LineRegress.fit(x[:15], y[:15])
print(regression_evaluator(result.predict, np.array(x[14:16]), np.array(y[14:16])))
