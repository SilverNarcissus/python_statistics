import math
from scipy.stats import f


class Solution:
    def solve(self):
        alpha = 0.1
        stand_x = math.sqrt(0.34)
        stand_y = math.sqrt(0.29)
        n_x = 18
        n_y = 13

        temp = math.pow(stand_x, 2) / math.pow(stand_y, 2)
        return [temp / f.ppf(alpha / 2, n_x - 1, n_y - 1), temp / f.ppf(1 - alpha / 2, n_x - 1, n_y - 1)]


s = Solution()
print(s.solve())
