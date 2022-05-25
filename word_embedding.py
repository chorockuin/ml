from gensim.models import Word2Vec
import gensim.downloader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

google_news = gensim.downloader.load('word2vec-google-news-300')

vocabs = ['thank', 'bye', 'dog', 'cat', 'animal', 'newspaper', 'magazine', 'king',
          'queen', 'princess', 'prince', 'paper', 'man', 'woman', 'men', 'women']

word_vectors_list = [google_news[v] for v in vocabs]

pca = PCA(n_components=2)
xy = pca.fit_transform(word_vectors_list)
x = xy[:, 0]
y = xy[:, 1]

plt.scatter(x, y, marker='o')
for i, v in enumerate(vocabs):
    plt.annotate(v, xy=(x[i], y[i]))
plt.show()