import numpy as np

from data_mining.my_classifier.nodes.node import Add, Mul, Sigmoid, LogicalLoss
from data_mining.my_classifier.nodes.base_node import Value


class LogisticRegression:
    def __init__(self, learning_rate=0.01, step=1000):
        self.step = step
        self.learning_rate = learning_rate
        self.x_holder = Value(None)
        self.a_holder = Value(None)
        self.b_holder = Value(None)
        self.y_holder = Value(None)

        res = Mul(self.x_holder, self.a_holder)
        res = Add(res, self.b_holder)
        self.out = Sigmoid(res)
        self.loss = LogicalLoss(self.out, self.y_holder)

    def fit(self, x, y):
        self.x_holder.val = x
        self.y_holder.val = y
        self.a_holder.val = np.random.randn(x.shape[1], 1)
        self.b_holder.val = np.random.randn(y.shape[1], 1)

        for i in range(self.step):
            self.a_holder.train(self.learning_rate)
            self.b_holder.train(self.learning_rate)
            self.loss.clear()
            if i % 50 == 0:
                print("step: ", i, " loss: ", self.loss.forward())

    def predict(self, X):
        self.x_holder.val = X
        return self.out.forward() > 0.5
