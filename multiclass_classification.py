import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
import util

x, y = make_classification(n_classes=3, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=101)

svm = LinearSVC(C=1.0, multi_class='ovr')
svm.fit(x, y)
util.plot_decision_boundary(svm, x, y)

svm = LinearSVC(C=1.0, multi_class='crammer_singer')
svm.fit(x, y)
util.plot_decision_boundary(svm, x, y)