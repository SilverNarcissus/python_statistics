import math
from scipy.stats import t


class Solution:
    def solve(self):
        alpha = 0.05
        # this means mean(Xi - Yi)
        mean = 3.2
        # this means sqrt(var(Xi - Yi))
        stand = math.sqrt(2.408)
        n = 10

        freedom = n - 1
        value = mean * math.sqrt(n) / stand
        # assume value is positive
        # one-tailed test
        # H0 means two totalities' mean aren't significantly different
        conclusion = value < -t.ppf(alpha, freedom)
        return [round(freedom, 2), value, conclusion]


s = Solution()
print(s.solve())
