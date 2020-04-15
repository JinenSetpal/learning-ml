import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ch02 import perceptron

# Initialize values
s = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases", "iris", "iris.data")
df = pd.read_csv(s, header=None, encoding="utf-8")
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values

# Plot the values
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")
plt.xlabel('sepal length [cm]') 
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# Fitting the object
ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of Updates")
plt.show()

# Running the model
ppn.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
