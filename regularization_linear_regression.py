import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(101)

x = np.random.randn(50, 2)
y = np.random.randn(50)

x, y = np.meshgrid(x, y)

z = 1.3 * x + 0.1 * y + 0.42 + np.random.randn(*x.shape)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.3)

lr = LinearRegression()
lr.fit(x_train, y_train)
pred_lr = lr.predict(x_test)

ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
pred_ridge = ridge.predict(x_test)

lasso = Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
pred_lasso = lasso.predict(x_test)

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(x_train, y_train)
pred_elastic = elastic.predict(x_test)

print('linear regression rmse:', np.sqrt(mean_squared_error(y_test, pred_lr)))
print('ridge regression rmse:', np.sqrt(mean_squared_error(y_test, pred_ridge)))
print('lasso rmse:', np.sqrt(mean_squared_error(y_test, pred_lasso)))
print('elastic net rmse:', np.sqrt(mean_squared_error(y_test, pred_elastic)))
