import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import adaline
import adalineSGD

# Initialize values
s = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases", "iris", "iris.data")
df = pd.read_csv(s, header=None, encoding="utf-8")
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values

# Plot Error Differentiation
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = adaline.Adaline(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(sum squared error)")
ax[0].set_title("Adaline - Learning rate 0.01")

ada2 = adaline.Adaline(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Sum squared error")
ax[1].set_title("Adaline - Learning rate 0.0001")
plt.show()

ada = adalineSGD.AdalineSGD(eta=0.01, n_iter=15, random_state=1)
x_std = np.copy(X)
x_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
x_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
X = x_std
ada.fit(X, y)

# Running the model
ada.plot_decision_regions(X, y, classifier=ada)
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.xlabel("Epochs")
plt.ylabel("Average Cost")
plt.tight_layout()
plt.show()
