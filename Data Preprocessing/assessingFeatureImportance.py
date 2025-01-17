from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
X, y = df.iloc[:, 1:].values, df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


forest.fit(X_train_std, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
