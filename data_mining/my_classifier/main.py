import numpy as np
from data_mining.my_classifier.nodes.node import Add, Mul, Sigmoid, LogicalLoss
from data_mining.my_classifier.logistic_regression import LogisticRegression

rate = 0.01
if __name__ == '__main__':
    x = np.array([[1, 2], [1, 2], [1, 2]])
    a = np.array([[1], [1]])
    b = np.array([[1], [1]])
    y = np.array([[0], [0], [0]])

    model = LogisticRegression()
    model.fit(x, y)
    print(model.predict(x))
