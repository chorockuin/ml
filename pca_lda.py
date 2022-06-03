import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_wine

wine = load_wine()

x = wine['data']
y = wine['target']
print(x.shape)

pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)
print(x_pca.shape)

plt.title('wine dataset with pca')
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
plt.show()

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x, y)
x_lda = lda.transform(x)

plt.title('wine dataset with lda')
plt.scatter(x_lda[:, 0], x_lda[:, 1], c=y)
plt.show()
