import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

x = np.random.randn(100, 2)
y = [0 for _ in range(100)]

x = np.concatenate([x, np.random.randn(20, 2) + np.array([5, 7])], axis=0)
y += [1 for _ in range(20)]

y = np.array(y)

print(f'y=0: {len(y[y==0])}')
print(f'y=1: {len(y[y==1])}')
plt.scatter(x[:,0], x[:,1], c=y)
plt.show()

rus = RandomUnderSampler()
x_res, y_res = rus.fit_resample(x, y)
print(f'under sampler y=0: {len(y_res[y_res==0])}')
print(f'under sampler y=1: {len(y_res[y_res==1])}')
plt.scatter(x_res[:, 0], x_res[:, 1], c=y_res)
plt.show()

ros = RandomOverSampler()
x_res, y_res = ros.fit_resample(x, y)
print(f'over sampler y=0: {len(y_res[y_res==0])}')
print(f'over sampler y=1: {len(y_res[y_res==1])}')
plt.scatter(x_res[:, 0], x_res[:, 1], c=y_res)
plt.show()