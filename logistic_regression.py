import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import util

dataset = load_iris()
data = dataset['data']

x = data[:, :2]
y = dataset['target']

lr = LogisticRegression()
lr.fit(x, y)

plt.figure(figsize=(6, 5))
util.plot_decision_boundary(lr, x, y)
