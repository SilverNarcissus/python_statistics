import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

from machine_learning.evaluator import classification_evaluator

# load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

s1 = np.arange(25, 50)
s2 = np.arange(75, 100)
X = X[y != 2]
X1 = np.append(X[25:50], X[75:100], axis=0)
y = y[y != 2]
y1 = np.append(y.take(s1), y.take(s2))

# learning begin
clf = LogisticRegression(C=1, penalty='l1', tol=1e-4)
clf.fit(X1, y1)
print(np.append(X[0:25], X[50:75], axis=0))
print(classification_evaluator(clf.predict, np.append(X[0:25], X[50:75], axis=0), np.append(y[0:25], y[50:75], axis=0)))
