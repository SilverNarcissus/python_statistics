import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression


def sample(arr, n):
    return np.random.permutation(arr)[0:int(n)]


def standardization(data):
    mean = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mean) / sigma


def classification_evaluator(predict_method, data, target):
    if len(data) != len(target):
        raise ValueError(
            "data length isn't equal to target length: data len:" + str(len(data)) + " target len:" + str(len(target)))
    # main part
    error_num = 0
    total_num = len(data)
    for i in range(total_num):
        data_temp = data[i].reshape(1, -1)
        error_num += int((predict_method(data_temp) != target[i]))

    error_rate = error_num / total_num
    return error_rate


def q3a():
    # question a
    df = pd.read_excel('data.xlsx', usecols=[6])
    avg_age = np.array(df)
    avg_age = avg_age.reshape((avg_age.shape[0]))
    print(stats.kstest(standardization(avg_age), 'norm'))


def q3b():
    df = pd.read_excel('data.xlsx', usecols=[1, 6])
    data = np.array(df)
    part = [[], [], [], [], []]
    for row in data:
        part[int(row[0]) - 1].append(row[1])

    for i in range(5):
        print('group: ' + str(i))
        print('p-value: ' + str(stats.kstest(standardization(np.array(part[i])), 'norm').pvalue))
        print('variances:' + str(np.var(np.array(part[i]))))
        print('--------------------------------------------------------')


def q3c():
    df = pd.read_excel('data.xlsx', usecols=[1, 6])
    data = np.array(df)
    part = [[], [], [], [], []]
    for row in data:
        part[int(row[0]) - 1].append(row[1])

    # print(stats.f_oneway(part[0], part[1], part[2], part[3], part[4]))

    fig, axes = plt.subplots(1, 1)
    sns.distplot(part[0], ax=axes, kde=True)
    sns.distplot(part[1], ax=axes, kde=True)
    sns.distplot(part[2], ax=axes, kde=True)
    sns.distplot(part[3], ax=axes, kde=True)
    sns.distplot(part[4], ax=axes, kde=True)
    plt.show()


def q4():
    for col in range(2, 5):
        df = pd.read_excel('data.xlsx', usecols=[1, col])
        data = np.array(df)
        part = [[], [], [], [], []]
        for row in data:
            part[int(row[0]) - 1].append(row[1])

        draw = np.array(pd.read_excel('data.xlsx', usecols=[4]))
        fig, axes = plt.subplots(1, 1)
        draw = draw.T
        print(draw)
        sns.distplot(draw[0], ax=axes, kde=True)
        plt.show()

        # for i in range(5):
        #     print('group: ' + str(i))
        #     print('p-value: ' + str(stats.kstest(standardization(np.array(part[i])), 'norm').pvalue))
        #     print('variances:' + str(np.var(np.array(part[i]))))
        #     print('--------------------------------------------------------')
        #
        # print('************************************************************')
        #
        # for i in range(5):
        #     part[i] = np.log(part[i])
        #     print('group: ' + str(i))
        #     print('p-value: ' + str(stats.kstest(standardization(np.array(part[i])), 'norm').pvalue))
        #     print('variances:' + str(np.var(np.array(part[i]))))
        #     print('--------------------------------------------------------')


def q5():
    for col in range(2, 5):
        df = pd.read_excel('data.xlsx', usecols=[1, col])
        data = np.array(df)
        part = [[], [], [], [], []]
        for row in data:
            part[int(row[0]) - 1].append(row[1])

        print(stats.kruskal(part[0], part[1], part[2], part[3], part[4]))
        fig, axes = plt.subplots(1, 1)
        sns.distplot(part[0], ax=axes, kde=True)
        sns.distplot(part[1], ax=axes, kde=True)
        sns.distplot(part[2], ax=axes, kde=True)
        sns.distplot(part[3], ax=axes, kde=True)
        sns.distplot(part[4], ax=axes, kde=True)
        plt.show()


def q6():
    df = pd.read_excel('data.xlsx', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    data = np.array(df)
    # build data
    count = [0, 0, 0, 0, 0]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for row in data:
        i = int(row[0]) - 1
        if count[i] < 10:
            x_test.append(row[1:])
            y_test.append(row[0])
            count[i] += 1
        else:
            x_train.append(row[1:])
            y_train.append(row[0])
    # train
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf.fit(x_train, y_train)
    # evaluate
    print('accuracy: ', classification_evaluator(clf.predict, x_test, y_test))


def q7():
    df = pd.read_excel('data.xlsx', usecols=[1, 6])
    data = np.array(df)
    total_num = data.shape[0]
    part = [[], [], [], [], []]
    for row in data:
        part[int(row[0]) - 1].append(row[1])
    # sample method 1
    n = total_num * 0.2 * 0.1
    sample_part = [[], [], [], [], []]
    statistic = []
    p_value = []
    for i in range(10):
        for j in range(5):
            sample_part[j] = sample(part[j], n)
        statistic_pair = stats.f_oneway(sample_part[0], sample_part[1], sample_part[2], sample_part[3], sample_part[4])
        statistic.append(statistic_pair[0])
        p_value.append(statistic_pair[1])
    print("----------------------method 1---------------------------")
    print('statistic mean: ', np.mean(statistic))
    print('statistic std: ', np.std(statistic))
    print('p_value mean: ', np.mean(p_value))
    print('p_value std: ', np.std(p_value))
    print("----------------------------------------------------------")
    statistic = []
    p_value = []
    # sample method 2
    for i in range(10):
        for j in range(5):
            sample_part[j] = sample(part[j], 0.1 * len(part[j]))
        statistic_pair = stats.f_oneway(sample_part[0], sample_part[1], sample_part[2], sample_part[3], sample_part[4])
        statistic.append(statistic_pair[0])
        p_value.append(statistic_pair[1])
    print("----------------------method 2---------------------------")
    print('statistic mean: ', np.mean(statistic))
    print('statistic std: ', np.std(statistic))
    print('p_value mean: ', np.mean(p_value))
    print('p_value std: ', np.std(p_value))
    print("----------------------------------------------------------")


if __name__ == '__main__':
    q6()
