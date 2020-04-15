from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv('/home/jinen/Downloads/wdbc.data', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1,
                                           solver='lbfgs'))
pipe_lr.fit(X_train, y_train)
print(pipe_lr.predict(X_test))
print('Accuracy: ', pipe_lr.score(X_test, y_test) * 100, "%", sep='')
