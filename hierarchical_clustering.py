import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons

def visualize(hc, hc_linkage, x):
    y_pred = hc.fit_predict(x)
    plt.scatter(x[y_pred==0, 0], x[y_pred==0, 1], color='red')
    plt.scatter(x[y_pred==1, 0], x[y_pred==1, 1], color='blue')
    plt.title(f'hierarchical clustering w/ {hc_linkage} linkage')
    plt.show()

x, y = make_moons(n_samples=200, noise=0.1)

hc_linkages = ['single', 'complete', 'average', 'ward']
for hc_linkage in hc_linkages:
    visualize(AgglomerativeClustering(n_clusters=2, linkage=hc_linkage), hc_linkage, x)
