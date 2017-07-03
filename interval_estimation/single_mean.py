import math
from scipy.stats import norm


class Solution:
    def solve(self):
        alpha = 0.05
        mean = 105.36
        stand = 10
        n = 25
        return [(norm.ppf(alpha / 2) * stand) / math.sqrt(n) + mean,
                -(norm.ppf(alpha / 2) * stand) / math.sqrt(n) + mean]


s = Solution()
print(s.solve())
