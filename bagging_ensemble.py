import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
import util

iris = load_iris(as_frame=True)

df = iris['data']
x = df[['sepal length (cm)', 'sepal width (cm)']]
y = iris['target']

# bootstrap or pasting
bc = BaggingClassifier(base_estimator=SVC(), n_estimators=10, n_jobs=-1)
bc.fit(x, y)

plt.title('bagging ensemble')
util.plot_decision_boundary(bc, x.values, y)
print(bc.estimators_)