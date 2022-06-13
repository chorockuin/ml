import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import util

dataset = load_iris()
data = dataset['data']

x = data[:, :2]
y = dataset['target']

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x, y)
plt.figure(figsize=(6, 5))
plt.title('knn classifier with k=1')
util.plot_decision_boundary(model, x, y)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x, y)
plt.figure(figsize=(6, 5))
plt.title('knn classifier with k=5')
util.plot_decision_boundary(model, x, y)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x, y)
plt.figure(figsize=(6, 5))
plt.title('knn classifier with k=7')
util.plot_decision_boundary(model, x, y)

model = KNeighborsClassifier(n_neighbors=11)
model.fit(x, y)
plt.figure(figsize=(6, 5))
plt.title('knn classifier with k=11')
util.plot_decision_boundary(model, x, y)

w = np.array([[1.0, 0.0],[0.0, 0.1]])
x_train = np.matmul(x, w)
model = KNeighborsClassifier(n_neighbors=7, p=1)
model.fit(x_train, y)
plt.figure(figsize=(6, 5))
plt.title('knn classifier with p=1')
util.plot_decision_boundary(model, x_train, y)

model = KNeighborsClassifier(n_neighbors=7, p=2)
model.fit(x_train, y)
plt.figure(figsize=(6, 5))
plt.title('knn classifier with p=2')
util.plot_decision_boundary(model, x_train, y)

model = KNeighborsClassifier(n_neighbors=7, algorithm='brute', metric='mahalanobis', metric_params={'VI': np.cov(x)})
model.fit(x_train, y)
plt.figure(figsize=(6, 5))
plt.title('knn classifier with mahalanobis distance')
util.plot_decision_boundary(model, x_train, y)
