from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import os


# noinspection PyShadowingNames
def train_input_fn(X_train, batch_size=8):
    X = X_train.copy()
    train_x, train_y = X, X.pop('MPG')
    ds = tf.data.Dataset.from_tensor_slices((dict(train_x, ), train_y))
    return ds.shuffle(100).repeat().batch(batch_size)


# noinspection PyShadowingNames
def eval_input_fn(X_test, batch_size=8):
    X = X_test.copy()
    train_x, train_y = X, X.pop('MPG')
    ds = tf.data.Dataset.from_tensor_slices((dict(train_x, ), train_y))
    return ds.batch(batch_size)


columns = ['MPG', 'Cylinders', 'Displacement',
           'Horsepower', 'Weight', 'Acceleration',
           'ModelYear', 'Origin']
df = pd.read_csv('/home/jinen/Downloads/auto-mpg.data', header=None, names=columns,
                 na_values='?', comment='\t',
                 sep=' ', skipinitialspace=True)

df = df.dropna()
df = df.reset_index(drop=True)

X_train, X_test = train_test_split(df, train_size=0.8)
train_stats = X_train.describe().transpose()
numeric_column_names = ['Cylinders', 'Displacement',
                        'Horsepower', 'Weight', 'Acceleration']

X_train_norm, X_test_norm = X_train.copy(), X_test.copy()
for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    X_train_norm.loc[:, col_name] = (X_train_norm.loc[:, col_name] - mean) / std
    X_test_norm.loc[:, col_name] = (X_test_norm.loc[:, col_name] - mean) / std

numeric_features = []
for col in numeric_column_names:
    numeric_features.append(tf.feature_column.numeric_column(key=col))

feature_year = tf.feature_column.numeric_column(key='ModelYear')
bucketized_features = [tf.feature_column.bucketized_column(source_column=feature_year, boundaries=[73, 76, 79]), ]

feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(key='Origin', vocabulary_list=[1, 2, 3])
categorical_indicator_features = [tf.feature_column.indicator_column(feature_origin), ]

X = train_input_fn(X_test_norm)
batch = next(iter(X))
print('Keys:', batch[0].keys())

print('Model Years:', batch[0]['ModelYear'])

feature_columns = (numeric_features +
                   bucketized_features +
                   categorical_indicator_features)

boosted_tree = tf.estimator.BoostedTreesRegressor(
    feature_columns=feature_columns,
    n_batches_per_layer=20,
    n_trees=200,
    model_dir='models/autompg_boostedtreeregressor/')

epochs = 1000
batch_size = 8
total_steps = epochs * int(np.ceil(len(X_train) / batch_size))

boosted_tree.train(input_fn=lambda: train_input_fn(X_train_norm, batch_size=batch_size),
                   steps=total_steps)
eval_results = boosted_tree.evaluate(lambda: eval_input_fn(X_test_norm, batch_size=batch_size))
print('Average Loss {:.4f}'.format(eval_results['average_loss']))

pred_res = boosted_tree.predict(input_fn=lambda: eval_input_fn(X_test_norm, batch_size=batch_size))
print(next(iter(pred_res)))
