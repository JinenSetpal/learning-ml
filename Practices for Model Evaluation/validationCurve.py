import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('/home/jinen/Downloads/wdbc.data', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state=1,
                                           solver='lbfgs',
                                           penalty='l2',
                                           max_iter=10000))

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_score, test_score = validation_curve(estimator=pipe_lr,
                                           X=X_train,
                                           y=y_train,
                                           param_name='logisticregression__C',
                                           param_range=param_range,
                                           cv=10)
train_mean = np.mean(train_score, axis=1)
train_std = np.std(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)
test_std = np.std(test_score, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training Accuracy')
plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 color='blue',
                 alpha=0.15)
plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation Accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 color='green',
                 alpha=0.15)

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.03])
plt.show()
