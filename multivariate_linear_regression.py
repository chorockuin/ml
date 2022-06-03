import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(101)
gx1 = np.random.randn(20)
gx2 = np.random.randn(20)

x1, x2 = np.meshgrid(gx1, gx2)
y = 1.3*x1 + 0.1*x2 + 0.42 + np.random.randn(*x1.shape)

lr = LinearRegression()
x = np.stack([np.reshape(x1, -1), np.reshape(x2, -1)], axis=1)
lr.fit(x, np.reshape(y, -1))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1, x2, y)

xx1, xx2 = np.meshgrid(np.linspace(min(gx1), max(gx1), 10), np.linspace(min(gx2), max(gx2), 10))
# yy = np.zeros_like(xx1)
yy_pred = lr.predict(np.stack([np.reshape(xx1, -1), np.reshape(xx2, -1)], axis=1))
yy_pred = np.reshape(yy_pred, (10, 10))
ax.plot_surface(xx1, xx2, yy_pred, color='r', alpha=0.5)
ax.set_xlabel('$x1$')
ax.set_ylabel('$x2$')
ax.set_zlabel('$y$')
plt.show()
