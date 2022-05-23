import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_boston

def visualize(tree, x, y):
    tree.fit(x, y)
    x_test = np.reshape(np.linspace(0.0, 40.0, 50), (-1, 1))
    pred_y = tree.predict(x_test)
    plt.figure(figsize=(6,5))
    plt.plot(x_test, pred_y, 'r-')
    plt.scatter(x[:, 0], y)
    plt.show()
    plt.figure(figsize=(20,12))
    plot_tree(tree)
    plt.show()

dataset = load_boston()
print(dataset.DESCR)

data = dataset['data']
print(data.shape)

x = np.reshape(data[:, 12], (-1, 1))
print(x.shape)

y = dataset['target']

visualize(DecisionTreeRegressor(), x, y)
visualize(DecisionTreeRegressor(max_depth=5, ccp_alpha=0.02), x, y)
