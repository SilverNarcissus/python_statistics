import math
from scipy.stats import norm, t


# condition: assume that two totalities variance are same
class Solution:
    def solve_n(self):
        alpha = 0.05
        beta = 0.05
        variance = 1
        delta = 1
        # two-tailed test
        return math.ceil((norm.ppf(alpha / 2) + norm.ppf(beta) * variance / delta) ** 2)

    def solve_beta(self):
        alpha = 0.05
        variance = 1
        delta = 0.75
        n = 30
        print(t.ppf(alpha, n - 1) / (delta / variance))
        return norm.cdf(t.ppf(alpha, n - 1) / (delta / variance))


s = Solution()
print(s.solve_n())
print(s.solve_beta())
