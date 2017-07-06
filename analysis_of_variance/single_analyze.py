from scipy.stats import f_oneway


def solve(data):
    freedom = len(data) * len(data[0]) - len(data)
    f, p_value = f_oneway(*data)
    print(f)
    if p_value < 0.01:
        return [freedom, f, "very high"]
    if p_value < 0.05:
        return [freedom, f, "high"]
    return [freedom, f, "no"]


# it means use 5 different methods and each method test 4 times
method_a = [25.6, 22.2, 28.0, 29.8]
method_b = [24.4, 30.0, 29.0, 27.5]
method_c = [25.0, 27.7, 23.0, 32.2]
method_d = [28.8, 28.0, 31.5, 25.9]
method_e = [20.6, 21.2, 22.0, 21.2]

print(solve((method_a, method_b, method_c, method_d, method_e)))
