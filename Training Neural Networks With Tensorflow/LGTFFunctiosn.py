# Linear Regressing Using Tensorflow Functions
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


class myModel(tf.keras.Model):
    def __init__(self):
        super(myModel, self).__init__()
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def call(self, inputs, training=None, mask=None):
        return self.w * inputs + self.b


X_train = np.arange(10).reshape((10, 1))
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0, 9.0])
tf.random.set_seed(1)
model = myModel()
model.compile(optimizer='sgd',
              loss=loss_fn,
              metrics=['mae', 'mse'])

model.fit(X_train_norm, y_train,
          epochs=200, batch_size=1,
          verbose=1)

X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 1, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training Examples', 'Linear Regression'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()
