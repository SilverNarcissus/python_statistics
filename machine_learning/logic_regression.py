import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from machine_learning.evaluator import bio_classification_evaluator
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.svm import NuSVC
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score

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
print(bio_classification_evaluator(clf.predict, np.append(X[0:25], X[50:75], axis=0),
                                   np.append(y[0:25], y[50:75], axis=0)))

# neural_network
# layer = []
# for i in range(6):
#     layer.append(100)
# layer_tuple = tuple(layer)
# result = MLPClassifier(hidden_layer_sizes=layer_tuple, max_iter=10000, tol=1e-8)
# # result.fit(iris.data, iris.target)
# scores = cross_val_score(result, iris.data, iris.target, cv=10)
# print(scores)

# # tree
# result = DecisionTreeClassifier(criterion="entropy", min_samples_split=7)
# # result.fit(iris.data, iris.target)
# scores = stats.describe(cross_val_score(result, iris.data, iris.target, cv=10))
# print(scores)

# svm
result = NuSVC(nu=0.4, probability=True, tol=1e-10, decision_function_shape="ovo")
# result.fit(iris.data, iris.target)
scores = stats.describe(cross_val_score(result, iris.data, iris.target, cv=10))
print(scores)

kmeans = KMeans(n_clusters=3).fit(iris.data)
print(kmeans.labels_)
print(fowlkes_mallows_score(iris.target, kmeans.labels_))
