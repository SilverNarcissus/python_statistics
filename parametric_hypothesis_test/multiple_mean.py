import math
from scipy.stats import norm, t


# condition: totality is norm or simple is larger than 30
class Solution:
    def solve(self):
        alpha = 0.05
        mean_x = 52.1
        mean_y = 27.1
        # if simple is larger than 30 s = sqrt(variance)
        stand_x = 45.1
        stand_y = 26.4
        # means the difference of two groups of data
        delta = 0
        n_x = 22
        n_Y = 22

        freedom = n_x - 1
        z = (mean_x - mean_y) / math.sqrt((math.pow(stand_x, 2) / n_x + math.pow(stand_y, 2) / n_Y))
        # assume z is positive
        # one-tailed test
        # H0 means two totalities' mean aren't significantly different
        conclusion = z < -norm.ppf(alpha)
        return [freedom, round(z, 2), conclusion]


s = Solution()
print(s.solve())
