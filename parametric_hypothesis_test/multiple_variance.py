import math
from scipy.stats import f


# condition: assume that two totalities variance are same
class Solution:
    def solve(self):
        alpha = 0.05
        stand_x = math.sqrt(0.34)
        stand_y = math.sqrt(0.29)
        n_x = 10
        n_y = 10

        value = math.pow(stand_x, 2) / math.pow(stand_y, 2)
        # one-tailed test
        # H0 means two totalities' variance aren't significantly different
        conclusion = value < f.ppf(1 - alpha, n_x - 1, n_y - 1)
        return [n_x - 1, value, conclusion]


s = Solution()
print(s.solve())