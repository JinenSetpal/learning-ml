import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_moons


def rbg_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = np.exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    alphas = np.column_stack([eigvecs[:, i]
                              for i in range(n_components)])
    lambdas = [eigvals[i] for i in range(n_components)]
    return alphas, lambdas


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbg_kernel_pca(X, gamma=15, n_components=1)
x_new = X[25]
x_proj = alphas[25]

x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
plt.scatter(alphas[y == 0, 0], np.zeros(50),
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros(50),
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
            label='original projection',
            marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
            label='new projection',
            marker='x', s=500)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.yticks([], [])
plt.legend(scatterpoints=1)
plt.tight_layout()
plt.show()
