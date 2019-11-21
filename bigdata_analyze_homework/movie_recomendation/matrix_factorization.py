import numpy as np
import matplotlib.pyplot as plt
from util import generate_data, eval_rmse

# 默认隐空间维度
k = 50
# 学习率
learning_rate = 0.0001
# 默认lambda
a = 0.001
# 用户数
user_num = 10000
# 电影数
movie_num = 10000

# grid search method
# k_grid = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# a_grid = [0.1, 0.05, 0.01, 0.005, 0.001]
k_grid = [5, 7, 9, 11, 13, 15]
a_grid = [0.008, 0.005, 0.003, 0.001]


def train_and_eval(train, test, k, a):
    # indicate which movie was scored
    A = train != 0
    # initialize
    U = np.random.normal(size=(user_num, k))
    V = np.random.normal(size=(movie_num, k))

    # for plot
    j_list = []
    rmse_list = []
    count = 0

    for i in range(2000):
        print('current iteration: ', i, ' current k: ', k, 'current a: ', a)
        M = A * (np.dot(U, V.T) - train)
        U_gradient = M.dot(V) + 2 * a * U
        V_gradient = M.T.dot(U) + 2 * a * V
        U = U - learning_rate * U_gradient
        V = V - learning_rate * V_gradient

        J = (np.linalg.norm(A * (train - np.dot(U, V.T))) ** 2) / 2 + \
            a * (np.linalg.norm(U) ** 2) + a * (np.linalg.norm(V) ** 2)
        final_score = np.dot(U, V.T)
        rmse = eval_rmse(test, final_score)

        print('J ', J)
        print('RMSE ', rmse)

        # test convergence
        if j_list and abs(J - j_list[len(j_list) - 1]) < 1:
            print("convergence， break")
            break

        j_list.append(J)
        rmse_list.append(rmse)
        count += 1

    return j_list, rmse_list, count


def q1():
    train, test = generate_data()
    j_list, rmse_list, count = train_and_eval(train, test, k, a)

    # J display
    plt.switch_backend('agg')
    plt.figure(figsize=(12, 7.6))
    ax1 = plt.subplot(1, 2, 1)
    plt.sca(ax1)
    plt.xlim(0, count)
    plt.xlabel('iteration')
    plt.ylabel('J')
    plt.plot(j_list)

    # RMSE display
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax2)
    plt.subplots_adjust(wspace=0.2)
    plt.xlim(0, count)
    plt.xlabel('iteration')
    plt.ylabel('RMSE')
    plt.plot(rmse_list)

    plt.savefig("result.png")


def q2():
    train, test = generate_data()
    with open('result2.txt', 'w', encoding='utf-8') as result:
        for k in k_grid:
            for a in a_grid:
                j_list, rmse_list, count = train_and_eval(train, test, k, a)
                result.write("current k: " + str(k)
                             + "current a: " + str(a)
                             + " RMSE result: " + str(rmse_list[len(rmse_list) - 1])
                             + " J result: " + str(j_list[len(j_list) - 1])
                             + '\n')


if __name__ == '__main__':
    q2()
