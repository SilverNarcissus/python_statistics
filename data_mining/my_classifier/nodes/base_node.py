class Node:
    def __init__(self):
        self.n1 = None
        self.n2 = None
        self.forward_value = None
        self.backward_value = None
        self.next = None
        self.next_backward_loc = None

    def forward(self):
        if self.forward_value is None:
            self.cal_forward()
        return self.forward_value

    def backward(self):
        if self.backward_value is None:
            self.cal_backward()

        return self.backward_value

    def cal_forward(self):
        pass

    def cal_backward(self):
        pass

    def clear(self):
        self.forward_value = None
        self.backward_value = None
        if self.n1 is not None:
            self.n1.clear()
        if self.n2 is not None:
            self.n2.clear()


class Value:
    def __init__(self, val):
        self.val = val
        self.n2 = None
        self.next = None
        self.next_backward_loc = None

    def forward(self):
        return self.val

    def backward(self):
        return self.next.backward()[self.next_backward_loc]

    def train(self, a):
        self.val = self.val - a * self.backward()

    def clear(self):
        pass
