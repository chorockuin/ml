import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import util

iris = load_iris(as_frame=True)

df = iris['data']
x = df[['sepal length (cm)', 'sepal width (cm)']]
y = iris['target']

tree = DecisionTreeClassifier()
tree.fit(x, y)

plt.title('decision tree')
util.plot_decision_boundary(tree, x.values, y)

rf = RandomForestClassifier()
rf.fit(x, y)

plt.title('random forest')
util.plot_decision_boundary(rf, x.values, y)
