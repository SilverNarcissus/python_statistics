import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

from machine_learning.evaluator import classification_evaluator


class OvRMultiClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, simple):
        possible = -1000000
        optimum = -1
        for i in range(len(self.classifiers)):
            if self.classifiers[i].decision_function(simple) > possible:
                possible = self.classifiers[i].decision_function(simple)
                optimum = i
            if self.classifiers[i].predict(simple) == 1:
                return i
        return optimum

    @staticmethod
    def fit(classifiers, array_x, array_y, num):
        # train N classifiers
        for i in range(num):
            y = np.copy(array_y)
            for j in range(len(y)):
                y[j] = (y[j] == i)
            classifiers[i].fit(array_x, y)

        return OvRMultiClassifier(classifiers)


# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

s1 = np.arange(10, 50)
s2 = np.arange(60, 100)
s3 = np.arange(110, 150)
X1 = np.append(X[10:50], X[60:100], axis=0)
X1 = np.append(X1, X[110:150], axis=0)
y1 = np.append(y.take(s1), y.take(s2))
y1 = np.append(y1, y.take(s3))

# learning begin
error_min = 1
optimum = -1
for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    clf = []
    for i in range(3):
        clf.append(LogisticRegression(C=j, penalty='l1', tol=1e-4))

    result = OvRMultiClassifier.fit(clf, X1, y1, 3)

    print(classification_evaluator(result.predict, np.append(np.append(X[0:10], X[50:60], axis=0), X[100:110], axis=0),
                                   np.append(np.append(y[0:10], y[50:60], axis=0), y[100:110], axis=0)))
    if classification_evaluator(result.predict, np.append(np.append(X[0:10], X[50:60], axis=0), X[100:110], axis=0),
                                np.append(np.append(y[0:10], y[50:60], axis=0), y[100:110], axis=0)) < error_min:
        error_min = classification_evaluator(result.predict,
                                             np.append(np.append(X[0:10], X[50:60], axis=0), X[100:110], axis=0),
                                             np.append(np.append(y[0:10], y[50:60], axis=0), y[100:110], axis=0))
        optimum = j
print(optimum, error_min)
