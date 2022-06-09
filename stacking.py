import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import load_iris
import util

iris = load_iris(as_frame=True)

df = iris['data']
x = df[['sepal length (cm)', 'sepal width (cm)']]
y = iris['target']

stacking = StackingClassifier([('lr', LogisticRegression()), ('dt', DecisionTreeClassifier()), ('svm', SVC())],
                              final_estimator=LogisticRegression())
stacking.fit(x, y)

plt.title('stacking classifier')
util.plot_decision_boundary(stacking, x.values, y)

plt.title('logistic classifier')
util.plot_decision_boundary(stacking.estimators_[0], x.values, y)

plt.title('decision tree')
util.plot_decision_boundary(stacking.estimators_[1], x.values, y)

plt.title('svm')
util.plot_decision_boundary(stacking.estimators_[2], x.values, y)
