import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import util

def visualize(tree, x, y):
    tree.fit(x, y)
    util.plot_decision_boundary(tree, x.values, y.values)
    plot_tree(tree, feature_names=['sepal length', 'sepal width'],
            class_names=iris.target_names,
            filled=True)
    plt.show()

iris = load_iris(as_frame=True)
df = iris['data']

x = df[['sepal length (cm)', 'sepal width (cm)']]
y = iris['target']

visualize(DecisionTreeClassifier(max_depth=5), x, y)
visualize(DecisionTreeClassifier(max_depth=5, ccp_alpha=0.01), x, y)