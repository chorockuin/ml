import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_circles

def visualize(x, y, inlier_index, outlier_index, title):
    plt.title(title)
    plt.scatter(x[y==inlier_index, 0], x[y==inlier_index, 1], color='red')
    plt.scatter(x[y==outlier_index, 0], x[y==outlier_index, 1], color='blue')
    plt.legend(['inliner', 'outlier'], loc='upper right')
    plt.show()

x, y = make_circles(n_samples=1500, factor=0.5, noise=0.02)
print(x.shape)
visualize(x, y, 1, 0, 'ground truth')

svm = OneClassSVM(kernel='linear')
svm.fit(x[y==1, :])
y_pred = svm.predict(x)
visualize(x, y_pred, -1, 1, 'one class linear svm')

svm = OneClassSVM(kernel='rbf', nu=0.01)
svm.fit(x[y==1, :])
y_pred = svm.predict(x)
visualize(x, y_pred, -1, 1, 'one class kernel svm w/ rbf kernel')

svm = OneClassSVM(kernel='poly')
svm.fit(x[y==1, :])
y_pred = svm.predict(x)
visualize(x, y_pred, -1, 1, 'one class kernel svm w/ poly kernel')

svm = OneClassSVM(kernel='sigmoid')
svm.fit(x[y==1, :])
y_pred = svm.predict(x)
visualize(x, y_pred, -1, 1, 'one class kernel svm w/ sigmoid kernel')

lof = LocalOutlierFactor(novelty=True)
lof.fit(x[y==1, :])
y_pred = lof.predict(x)
visualize(x, y_pred, -1, 1, 'local outlier factdor')
