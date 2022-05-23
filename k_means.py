import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

k = 10
x, y = make_blobs(n_samples=1000, centers=k, n_features=2, random_state=101)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

k_means = KMeans(n_clusters=k)
k_means.fit(x)

y_test = k_means.predict(x)
cents = k_means.cluster_centers_

plt.scatter(x[:, 0], x[:, 1], c=y_test)
plt.scatter(cents[:, 0], cents[:, 1], marker='o', c='r')
plt.show()

dists = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(x)
    dists.append(k_means.inertia_)

print(dists)
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), dists)
plt.grid(True)
plt.title('elbow curve')
plt.show()