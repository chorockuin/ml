import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles, make_moons

x, y = make_moons()

plt.title('moon dataset')
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)

plt.title('moon dataset with pca')
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.show()

k_pca = KernelPCA(kernel='rbf', gamma=20)
k_pca.fit(x, y)
x_k_pca = k_pca.transform(x)

plt.title('moon dataset with kernel pca')
plt.scatter(x_k_pca[:, 0], x_k_pca[:, 1], c=y)
plt.show()