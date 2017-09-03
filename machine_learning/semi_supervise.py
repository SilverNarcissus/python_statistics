from scipy.stats import stats
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.semi_supervised import LabelPropagation
import numpy as np

label_prop_model = LabelPropagation(kernel="knn")

iris = datasets.load_iris()
rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.9
print(random_unlabeled_points)
labels = np.copy(iris.target)
labels[random_unlabeled_points] = -1

label_prop_model.fit(iris.data, labels)
print(label_prop_model.predict(iris.data))

# scores = stats.describe(cross_val_score(label_prop_model, iris.data, labels, cv=2))
# print(scores)
