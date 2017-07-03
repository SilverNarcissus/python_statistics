import math
from scipy.stats import t


class Solution:
    def solve(self):
        alpha = 0.05
        mean_x = 500
        mean_y = 496
        stand_x = 1.10
        stand_y = 1.2
        n_x = 10
        n_y = 20

        freedom = n_x + n_y - 2
        sw = math.sqrt(((n_x - 1) * math.pow(stand_x, 2) + (n_y - 1) * math.pow(stand_y, 2)) / freedom)
        temp = math.sqrt(1 / n_x + 1 / n_y)
        return [mean_x - mean_y + t.ppf(alpha / 2, freedom) * sw * temp,
                mean_x - mean_y - t.ppf(alpha / 2, freedom) * sw * temp]


s = Solution()
print(s.solve())
