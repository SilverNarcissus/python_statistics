from data_mining.my_classifier.nodes.base_node import Node
import numpy as np


class Add(Node):
    def __init__(self, n1, n2):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        n1.next = self
        n2.next = self
        n1.next_backward_loc = 0
        n2.next_backward_loc = 1

    def cal_forward(self):
        self.forward_value = self.n1.forward() + self.n2.forward()

    def cal_backward(self):
        next_backward = 1 if self.next is None else self.next.backward()[self.next_backward_loc]
        self.backward_value = [next_backward, next_backward]


class Mul(Node):
    def __init__(self, n1, n2):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        n1.next = self
        n2.next = self
        n1.next_backward_loc = 0
        n2.next_backward_loc = 1

    def cal_forward(self):
        self.forward_value = self.n1.forward().dot(self.n2.forward())

    def cal_backward(self):
        next_backward = 1 if self.next is None else self.next.backward()[self.next_backward_loc]
        self.backward_value = [next_backward.dot(self.n2.forward().T), self.n1.forward().T.dot(next_backward)]


class Sigmoid(Node):
    def __init__(self, n1):
        super().__init__()
        self.n1 = n1
        n1.next = self
        n1.next_backward_loc = 0

    def cal_forward(self):
        self.forward_value = 1 / (1 + np.exp(- self.n1.forward()))

    def cal_backward(self):
        next_backward = 1 if self.next is None else self.next.backward()[self.next_backward_loc]
        sigmoid = 1 / (1 + np.exp(- self.n1.forward()))
        self.backward_value = [next_backward * sigmoid * (1 - sigmoid)]


class LogicalLoss(Node):
    def __init__(self, predict, label):
        super().__init__()
        self.n1 = predict
        self.n2 = label
        predict.next = self
        label.next = self
        predict.next_backward_loc = 0
        label.next_backward_loc = 1

    def cal_forward(self):
        self.forward_value = - self.n2.forward() * np.log2(self.n1.forward()) \
                             - (1 - self.n2.forward()) * np.log2(1 - self.n1.forward())

    def cal_backward(self):
        next_backward = 1 if self.next is None else self.next.backward()[self.next_backward_loc]
        predict_backward = next_backward * ((-self.n2.forward()) * (1 / -self.n1.forward())
                                            + (1 - self.n2.forward()) * (1 / (1 - self.n1.forward())))
        label_backward = next_backward * (- np.log2(self.n1.forward()) + np.log2(1 - self.n1.forward()))
        self.backward_value = [predict_backward, label_backward]
