import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
import util

def visualize(svm, k, c):
    svm.fit(x, y)
    plt.title(f'{k} kernel svm w/ C={c}')
    util.plot_decision_boundary(svm, x, y)

x, y = make_moons(n_samples=100, random_state=1)

params = [{'k': 'linear', 'c': 1.0}, 
          {'k': 'poly', 'c': 0.25},
          {'k': 'rbf', 'c': 1.0},
          {'k': 'sigmoid', 'c': 1.0}]
for p in params:
    visualize(SVC(kernel=p['k'], C=p['c']), p['k'], p['c'])

x, y = make_circles(n_samples=100, random_state=1)

for p in params:
    visualize(SVC(kernel=p['k'], C=p['c']), p['k'], p['c'])