import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

x, y = make_regression(n_samples=100, n_features=1, noise=5.0, random_state=101)

lr = LinearRegression()
lr.fit(x, y)
print(lr.coef_, lr.intercept_)

x_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = lr.predict(x_test)

plt.figure()
plt.scatter(x, y)
plt.plot(x_test, y_pred, 'r-')
plt.show()
