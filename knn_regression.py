import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor

dataset = load_boston()
data = dataset['data']

x = np.reshape(data[:, 12], (-1, 1))
y = dataset['target']

model = KNeighborsRegressor(n_neighbors=11)
model.fit(x, y)

plt.figure(figsize=(6, 5))
plt.scatter(x[:, 0], y)

x_test = np.reshape(np.linspace(0.0, 40.0, 50), (-1, 1))
pred_y = model.predict(x_test)

plt.plot(x_test, pred_y, 'r-')