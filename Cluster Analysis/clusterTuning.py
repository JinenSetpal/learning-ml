from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                n_init=10,
                n_jobs=2,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11),
         distortions,
         marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortions')
plt.tight_layout()
plt.show()
