import plac
import os

import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import ward, dendrogram
from mpl_toolkits.mplot3d import Axes3D


def main():
    filenames = ['data/austen-brontë/Austen_Emma.txt', 'data/austen-brontë/Austen_Pride.txt',
                 'data/austen-brontë/Austen_Sense.txt', 'data/austen-brontë/CBronte_Jane.txt',
                 'data/austen-brontë/CBronte_Professor.txt', 'data/austen-brontë/CBronte_Villette.txt']
    vectorizer = CountVectorizer(input='filename')
    dtm = vectorizer.fit_transform(filenames)  # a sparse matrix
    vocab = vectorizer.get_feature_names()  # a list

    dtm = dtm.toarray()  # convert to a regular array
    vocab = np.array(vocab)

    # use the standard Python list method index(...)
    # list(vocab) or vocab.tolist() will take vocab (an array) and return a list
    house_idx = list(vocab).index('house')
    print('occurrences:', dtm[0, house_idx], dtm[0, vocab == 'house'])

    # "by hand"
    n, _ = dtm.shape
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x, y = dtm[i, :], dtm[j, :]
            dist[i, j] = np.sqrt(np.sum((x - y) ** 2))

    dist = euclidean_distances(dtm)
    print('\n1:', np.round(dist, 1))

    # the distance between *Pride and Prejudice* and *Jane Eyre*
    print('\n2:', dist[1, 3])

    # which is greater than the distance between *Jane Eyre* and *Villette* (index 5)
    # (dist[1, 3] > dist[3, 5]) == True

    dist = 1 - cosine_similarity(dtm)

    print('\n4:', np.round(dist, 2))

    # the distance between *Pride and Prejudice* (index 1)
    # and *Jane Eyre* (index 3) is
    print('\n5:', dist[1, 3])

    # which is greater than the distance between *Jane Eyre* and
    # *Villette* (index 5)
    # (dist[1, 3] > dist[3, 5]) == True

    norms = np.sqrt(np.sum(dtm * dtm, axis=1, keepdims=True))  # multiplication between arrays is element-wise
    dtm_normed = dtm / norms
    similarities = np.dot(dtm_normed, dtm_normed.T)

    print('\n7:', np.round(similarities, 2))

    # similarities between *Pride and Prejudice* and *Jane Eyre* is
    print('\n8:', similarities[1, 3])

    """ 2D Graph """

    # two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]

    # short versions of filenames:
    # convert 'data/austen-brontë/Austen_Emma.txt' to 'Austen_Emma'
    names = [os.path.basename(fn).replace('.txt', '') for fn in filenames]

    # color-blind-friendly palette
    for x, y, name in zip(xs, ys, names):
        color = 'orange' if "Austen" in name else 'skyblue'
        plt.scatter(x, y, c=color)
        plt.text(x, y, name)

    plt.show()

    """ 3D Graph """

    # après Jeremy M. Stober, Tim Vieira
    # https://github.com/timvieira/viz/blob/master/mds.py
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print('x:', pos[:, 0])
    print('y:', pos[:, 1])
    print('z:', pos[:, 2])

    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    print('\nscatter:', scatter)

    for x, y, z, s in zip(pos[:, 0], pos[:, 1], pos[:, 2], names):
        ax.text(x, y, z, s)

    plt.show()

    """ tree """

    linkage_matrix = ward(dist)

    # match dendrogram to that returned by R's hclust()
    cluster = dendrogram(linkage_matrix, orientation="right", labels=names)
    print('\ncluster:', cluster)

    plt.tight_layout()  # fixes margins

    plt.show()


if __name__ == '__main__':
    try:
        plac.call(main)
    except KeyboardInterrupt:
        print('\nGoodbye!')
