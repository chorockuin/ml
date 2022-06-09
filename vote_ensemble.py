import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris
import util

iris = load_iris(as_frame=True)

df = iris['data']
x = df[['sepal length (cm)', 'sepal width (cm)']].values
y = iris['target']

mv = VotingClassifier([('lr', LogisticRegression()), ('dt', DecisionTreeClassifier()), ('svm', SVC())],
                      voting='hard',
                      n_jobs=-1)
mv.fit(x, y)

plt.title('voting ensemble')
util.plot_decision_boundary(mv, x, y)

plt.title('logistic regression')
util.plot_decision_boundary(mv.estimators_[0], x, y)

plt.title('decision tree')
util.plot_decision_boundary(mv.estimators_[1], x, y)

plt.title('kernel SVM')
util.plot_decision_boundary(mv.estimators_[2], x, y)
