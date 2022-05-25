import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_moons

markers = ['o', 'x']
colors = ['blue', 'red']

def visualize(x, title):
    plt.title(title)
    for i, mc in enumerate(zip(markers, colors)):
        plt.scatter(x[y==i, 0], x[y==i, 1], marker=mc[0], color=mc[1], alpha=0.8)
    plt.show()

x, y = make_moons(n_samples=1000, noise=0.1)

visualize(x, 'moon dataset')

pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)
print(x_pca.shape)

visualize(x_pca, 'pca')

k_pca = KernelPCA(kernel='rbf', gamma=15.0)
k_pca.fit(x)
x_k_pca = k_pca.transform(x)

visualize(x_k_pca, 'kernel_pca')

t_sne = TSNE(n_components=2)
x_t_sne = t_sne.fit_transform(x)

visualize(x_t_sne, 't_sne')