import math
from scipy.stats import t


# condition: two totalities variance are same
class Solution:
    def solve(self):
        alpha = 0.05
        mean_x = 79.43
        mean_y = 76.23
        stand_x = math.sqrt(2.225)
        stand_y = math.sqrt(3.325)
        # means the difference of two groups of data
        delta = 0
        n_x = 10
        n_y = 10

        freedom = n_x + n_y - 2
        sw = math.sqrt(((n_x - 1) * math.pow(stand_x, 2) + (n_y - 1) * math.pow(stand_y, 2)) / freedom)
        temp = math.sqrt(1 / n_x + 1 / n_y)
        value = (mean_x - mean_y - delta) / (sw * temp)
        # assume value is positive
        # one-tailed test
        # H0 means two totalities' mean aren't significantly different
        conclusion = value < -t.ppf(alpha,freedom)
        return [freedom, value, conclusion]


s = Solution()
print(s.solve())
