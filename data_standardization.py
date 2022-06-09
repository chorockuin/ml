import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def show_xys(xys):
    print('mean: ', np.average(xys[:, 0]), np.average(xys[:, 1]))
    print('std: ', np.std(xys[:, 0]), np.std(xys[:, 1]))

    plt.scatter(xys[:, 0], xys[:, 1])
    plt.scatter(np.average(xys[:, 0]), np.average(xys[:, 1]), color=r'red')
    plt.axis('equal')
    plt.grid()
    plt.show()

xys = np.random.randn(100, 2)
show_xys(xys)

scaler = StandardScaler()
xys_scale = scaler.fit_transform(xys)
show_xys(xys_scale)
