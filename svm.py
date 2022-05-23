import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification, make_moons
import util

def visualize(x, y, slack_variables_c):
    svm = LinearSVC(C=slack_variables_c)
    svm.fit(x, y)
    plt.title(f'svm classification w/ C={slack_variables_c}')
    util.plot_decision_boundary(svm, x, y)

x, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.3,
    n_classes=2,
    random_state=1)

visualize(x, y, 1.0)
visualize(x, y, 0.2)
visualize(x, y, 0.01)

x, y = make_moons(n_samples=100, random_state=1)

visualize(x, y, 1.0)
visualize(x, y, 0.01)
