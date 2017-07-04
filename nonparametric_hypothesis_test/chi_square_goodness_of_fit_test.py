from scipy.stats import chi2


def solve(array_n, array_p):
    k = len(array_n)
    value = 0
    total = sum(array_n)
    for i in range(k):
        value += array_n[i] ** 2 / (total * array_p[i])
    value = value - total
    freedom = k - 1
    conclusion = value < chi2.ppf(0.05, freedom)
    return [round(freedom, 2), round(value, 2), conclusion]


array_n = [6, 16, 17, 26, 11, 9, 9, 6]
array_p = [0.078, 0.132, 0.185, 0.194, 0.163, 0.114, 0.069, 0.065]
print(solve(array_n, array_p))
