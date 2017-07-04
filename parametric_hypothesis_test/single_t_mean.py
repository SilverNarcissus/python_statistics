import math
from scipy.stats import t


class Solution:
    def solve(self):
        alpha = 0.05
        mean = 241.5
        u0 = 225
        stand = 98.7259
        n = 16

        freedom = n - 1
        z = (mean - u0) * math.sqrt(n) / stand
        # assume z is positive
        # one-tailed test
        # H0 means mean and u0 aren't significantly different
        conclusion = z < -t.ppf(alpha, freedom)
        return [round(freedom, 2), round(z, 2), conclusion]


s = Solution()
print(s.solve())
