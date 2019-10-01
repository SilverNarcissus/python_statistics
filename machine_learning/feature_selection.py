from random import random

import numpy as np
from scipy.stats import stats
from sklearn.decomposition import PCA, KernelPCA
from sklearn import datasets, feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.decomposition import DictionaryLearning
from sklearn.pipeline import Pipeline

iris = datasets.load_iris()
print(iris.data)

# pca = PCA(n_components=4)
# X = pca.fit_transform(iris.data)
# kpca = KernelPCA(n_components=2, kernel="linear")
# X = kpca.fit_transform(iris.data)
# print(X)
#
# # result = NuSVC(nu=0.4, probability=True, tol=1e-6, decision_function_shape="ovo")
# #result = LogisticRegression(C=1, penalty='l1', tol=1e-4)
# result = KNeighborsClassifier(n_neighbors=2)
# # result = DecisionTreeClassifier(criterion="entropy", min_samples_split=7)
# # result.fit(iris.data, iris.target)
# x = []
# for i in range(len(iris.data)):
#     x.append([iris.data[i][3]])
#
# # print(x)
# scores = stats.describe(cross_val_score(result, X, iris.target, cv=10))
# print(scores)
#
# # feature selection
# X, y = iris.data, iris.target
# # selector = VarianceThreshold(0.5)
# selector = feature_selection.SelectFromModel(result, threshold="median")
# #X = selector.fit_transform(X, y)
# print(X)
# X_new = SelectKBest(k=1).fit_transform(X, y)
# print(X_new)
# print(X)
dl = DictionaryLearning(3, transform_algorithm='lars')
pca = PCA(n_components=2)
# X = pca.fit_transform(iris.data)
X = pca.fit_transform(iris.data)
print(pca.transform([5, 3, 1.4, 0.2]))
print(iris.data[0])
print(X)
result = KNeighborsClassifier(n_neighbors=2, n_jobs=1)
s = NuSVC()
s.fit()

scores = stats.describe(cross_val_score(result, X, iris.target, cv=10))
result.fit(X, iris.target)
print(result.predict(pca.transform([5, 3, 1.4, 0.1])))
print(scores)

# final_estimator = Pipeline([('pca', pca), ('classifier', result)])
# scores = stats.describe(cross_val_score(final_estimator, iris.data, iris.target, cv=30))
# print(scores)

