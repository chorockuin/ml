import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
import util

iris = load_iris(as_frame=True)

df = iris['data']
x = df[['sepal length (cm)', 'sepal width (cm)']]
y = iris['target']

model = GradientBoostingClassifier()
model.fit(x, y)

plt.title('gradient boosting')
util.plot_decision_boundary(model, x.values, y)

model = XGBClassifier()
model.fit(x, y)

plt.title('xgb classifier')
util.plot_decision_boundary(model, x.values, y)
