import tensorflow as tf
import numpy as np


# noinspection PyShadowingNames
def input_train(x_train, y_train, batch_size=8):
    ds = tf.data.Dataset.from_tensor_slices(({'input_features': x_train}, y_train.reshape(-1, 1)))
    return ds.shuffle(100).repeat().batch(batch_size)


# noinspection PyShadowingNames
def eval_input(x_test, y_test=None, batch_size=8):
    if y_test is None:
        ds = tf.data.Dataset.from_tensor_slices(({'input_features': x_test}))
    else:
        ds = tf.data.Dataset.from_tensor_slices(({'input_features': x_test}, y_test.reshape(-1, 1)))
    return ds.batch(batch_size)


tf.random.set_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_test = x[100:, :]
y_test = y[100:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,), name='input_features'),
    tf.keras.layers.Dense(units=4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
])
print(model.summary())

features = [tf.feature_column.numeric_column(key='input_features:', shape=(2,))]
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

epochs = 200
batch_size = 2
steps = np.ceil(len(x_train) / batch_size)

estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='models/xor_estimator')
estimator.train(input_fn=lambda: input_train(x_train, y_train, batch_size),
                steps=epochs * steps)

model_eval = estimator.evaluate(input_fn=lambda: eval_input(x_test, y_test, batch_size))
print(model_eval)
