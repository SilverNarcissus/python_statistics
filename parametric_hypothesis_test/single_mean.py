import math
from scipy.stats import norm
from math import sqrt


class Solution:
    def solve(self):
        alpha = 0.01
        mean = 1.1
        u0 = 0
        stand = 4.9
        n = 51

        freedom = n - 1
        z = (mean - u0) * math.sqrt(n) /stand
        # one-tailed test
        conclusion = z < -norm.ppf(alpha)
        return [round(freedom, 2), round(z, 2), conclusion]


s = Solution()
print(s.solve())
