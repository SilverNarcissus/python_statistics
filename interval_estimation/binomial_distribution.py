import math
from scipy.stats import norm


class Solution:
    def solve(self):
        alpha = 0.05
        # mean = effective points / total points
        mean = 0.6
        n = 100

        a = n + math.pow(norm.ppf(alpha / 2), 2)
        b = -(2 * n * mean + math.pow(norm.ppf(alpha / 2), 2))
        c = n * math.pow(mean, 2)

        return [(-b - math.sqrt(math.pow(b, 2) - 4 * a * c)) / (2 * a),
                (-b + math.sqrt(math.pow(b, 2) - 4 * a * c)) / (2 * a)]


s = Solution()
print(s.solve())
