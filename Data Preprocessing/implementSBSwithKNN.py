import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from ch04.sequentialBackwardSelection import SBS

df = pd.read_csv(r'/home/jinen/Downloads/wine.data', header=None)
df.columns = ['Class label', 'Alcohol',
              'Malic Acid', 'Ash',
              'Alcalinity of Ash', 'Magnesium',
              'Total phenols', 'Flavanoids',
              'Nonflavanoid phenols',
              'Proanthocyanins',
              'Color intensity', 'Hue',
              'OD280/OD315 of diluted wines',
              'Proline']

X, y = df.iloc[:, 1:].values, df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.ylim([0.7, 1.02])
plt.grid()
plt.tight_layout()
plt.show()

print(df.columns[1:][list(sbs.subsets_[10])])

# Original Dataset
knn.fit(X_train_std, y_train)
print('Training Accuracy:', knn.score(X_train_std, y_train))
print('Testing Accuracy:', knn.score(X_test_std, y_test))

# Dataset with feature selection
knn.fit(X_train_std[:, list(sbs.subsets_[10])], y_train)
print('Using Feature Selection:')
print('Training Accuracy:', knn.score(X_train_std[:, list(sbs.subsets_[10])], y_train))
print('Testing Accuracy:', knn.score(X_test_std[:, list(sbs.subsets_[10])], y_test))
