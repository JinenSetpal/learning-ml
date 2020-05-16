from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)
km = KMeans(n_clusters=2,
            n_jobs=2,
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')

y_km = km.fit_predict(X)
print(y_km)

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.tight_layout()
plt.show()

y_ac = ac.fit_predict(X)
print(y_ac)

plt.scatter(X[y_ac == 0, 0],
            X[y_ac == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_ac == 1, 0],
            X[y_ac == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.tight_layout()
plt.show()
