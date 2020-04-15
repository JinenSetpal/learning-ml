import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print(X_test)
