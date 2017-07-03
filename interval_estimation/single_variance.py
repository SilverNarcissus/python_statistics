import math
from scipy.stats import chi2


class Solution:
    def solve(self):
        alpha = 0.05
        stand = math.sqrt(93.21)
        n = 25
        # chi2.ppf sees alpha as lower division not upper division
        return [(n - 1) * math.pow(stand, 2) / chi2.ppf(1 - alpha / 2, n - 1),
                (n - 1) * math.pow(stand, 2) / chi2.ppf(alpha / 2, n - 1)]


s = Solution()
print(s.solve())
