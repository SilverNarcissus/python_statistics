from scipy import stats


# predict_method: what you get in learning
# data: the data to predict
# target: the expectation value
# beta: use to caculate F1
#       beta > 1 means recall is more important
#       beta < 1 means precision is more important
def classification_evaluator(predict_method, data, target, beta=1):
    # parameter check
    if len(data) != len(target):
        raise ValueError(
            "data length isn't equal to target length: data len:" + str(len(data)) + " target len:" + str(len(target)))
    # main part
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total_num = len(data)

    for i in range(total_num):
        data_temp = data[i].reshape(1, -1)
        if target[i] == 1:
            tp += int((predict_method(data_temp) == target[i]))
            fn += int((predict_method(data_temp) != target[i]))
        else:
            tn += int((predict_method(data_temp) == target[i]))
            fp += int((predict_method(data_temp) != target[i]))

    precision = tp / (tp + fp)  # 查准率
    recall = tp / (tp + fn)  # 查全率
    error_rate = (fp + fn) / total_num  # 错误率
    F1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
    TPR = tp / (tp + fn)  # true positive rate
    FPR = fp / (tn + fp)  # false positive rate

    return [error_rate, precision, recall, F1, TPR, FPR]


# predict_method: what you get in learning
# data: the data to predict
# target: the expectation value
def regression_evaluator(predict_method, data, target):
    # parameter check
    if len(data) != len(target):
        raise ValueError(
            "data length isn't equal to target length: data len:" + str(len(data)) + " target len:" + str(len(target)))
    # main part
    total_num = len(data)
    error = []
    for i in range(total_num):
        data_temp = data[i].reshape(1, -1)
        error.append(abs(predict_method(data_temp) - target[i]))

    return stats.describe(error)
