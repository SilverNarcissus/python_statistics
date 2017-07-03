import math
from scipy.stats import norm


class Solution:
    def solve(self):
        alpha = 0.05
        mean_x = 86
        mean_y = 78
        # when the sample size larger than 30, the square of the simple standard deviation can be used as variance
        variance_x = 5.8 * 5.8
        variance_y = 7.2 * 7.2
        n_x = 46
        n_y = 33
        return [mean_x - mean_y + norm.ppf(alpha / 2) * math.sqrt(variance_x / n_x + variance_y / n_y),
                mean_x - mean_y - norm.ppf(alpha / 2) * math.sqrt(variance_x / n_x + variance_y / n_y)]


s = Solution()
print(s.solve())
