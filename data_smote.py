import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

x = np.random.randn(100, 2)
y = [0 for _ in range(100)]
x = np.concatenate([x, np.random.randn(20, 2) + np.array([5, 7])], axis=0)
y += [1 for _ in range(20)]

plt.scatter(x[:,0], x[:,1], c=y)
plt.title('original dataset')
plt.show()

smote = SMOTE()
x_res, y_res = smote.fit_resample(x, y)
plt.scatter(x_res[:, 0], x_res[:, 1], c=y_res)
plt.title('over sampled by SMOTE')
plt.show()
