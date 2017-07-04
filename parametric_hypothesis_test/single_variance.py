import math
from scipy.stats import norm, chi2
from math import sqrt


class Solution:
    def solve(self):
        alpha = 0.02
        # the simple standard deviation
        stand = math.sqrt(9200)
        # the totality variance
        variance = 5000
        n = 26

        freedom = n - 1
        ka2 = freedom * (stand ** 2) / variance
        # two-tailed test
        # H0 means simple is significantly fluctuate than totality
        conclusion = (ka2 > chi2.ppf(1 - alpha / 2, freedom)) | (ka2 > chi2.ppf(alpha / 2, freedom))
        return [round(freedom, 2), round(ka2, 2), conclusion]


s = Solution()
print(s.solve())
