import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
import util

iris = load_iris(as_frame=True)

df = iris['data']
x = df[['sepal length (cm)', 'sepal width (cm)']]
y = iris['target']

model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
model.fit(x, y)

plt.title('adaboost w/ decision tree weak learner')
util.plot_decision_boundary(model, x.values, y)

model = AdaBoostClassifier(base_estimator=LogisticRegression())
model.fit(x, y)

plt.title('adaboost w/ logistic regression weak learner')
util.plot_decision_boundary(model, x.values, y)
