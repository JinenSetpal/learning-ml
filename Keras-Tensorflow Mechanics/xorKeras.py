from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class myModel(tf.keras.Model):
    def __init__(self):
        super(myModel, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(
            units=4, activation=tf.keras.activations.relu)
        self.hidden_2 = tf.keras.layers.Dense(
            units=4, activation=tf.keras.activations.relu)
        self.hidden_3 = tf.keras.layers.Dense(
            units=4, activation=tf.keras.activations.relu)
        self.output_layer = tf.keras.layers.Dense(
            units=1, activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        h = self.hidden_1(inputs)
        h = self.hidden_2(h)
        h = self.hidden_3(h)
        h = self.hidden_3(h)
        return self.output_layer(h)


tf.random.set_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

x_train = x[:100, :]
y_train = y[:100]
x_valid = x[100:, :]
y_valid = y[100:]

tf.random.set_seed(1)
model = myModel()
model.build(input_shape=(None, 2))
model.summary()

model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.Accuracy()])

hist = model.fit(x_train, y_train,
                 validation_data=(x_valid, y_valid),
                 epochs=200, batch=2, verbose=0)
hist = hist.history

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 3, 1)
plt.plot(hist['loss'], lw=4)
plt.plot(hist['val_loss'], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(hist['accuracy'], lw=4)
plt.plot(hist['val_accuracy'], lw=4)
plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()
