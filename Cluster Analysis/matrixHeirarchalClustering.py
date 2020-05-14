from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = [1, 2, 3, 4, 5]
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels, index=labels)
print(row_dist)

row_clusters = linkage(df.values,
                       method='complete',
                       metric='euclidean')
print(pd.DataFrame(row_clusters,
                   columns=['row label 1',
                            'row label 2',
                            'distance',
                            'no. of items in clust.'],
                   index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])]))

row_dendr = dendrogram(row_clusters,
                       labels=labels)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
# noinspection PyRedeclaration
row_dendr = dendrogram(row_clusters,
                       orientation='left')
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
                  interpolation='nearest',
                  cmap='hot_r')

axd.set_xticks([])
axd.set_yticks([])
[i.set_visible(False) for i in axd.spines.values()]
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()
