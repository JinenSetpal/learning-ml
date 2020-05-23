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
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0, 9.0])

plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
ds_train_orig = tf.data.Dataset.from_tensor_slices((tf.cast(X_train_norm, tf.float32),
                                                    tf.cast(y_train, tf.float32)))
model = myModel()
model.build(input_shape=(None, 1))
model.summary()

tf.random.set_seed(1)
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train)) / batch_size)

ds_train = ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size)
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size)
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size)
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size)
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size)
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size)
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train)).repeat(count=None).batch(batch_size)
Ws, bs = [], []

for i, batch in enumerate(ds_train):
    if i >= steps_per_epoch * num_epochs:
        break
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())

    bx, by = batch
    loss_val = loss_fn(model(bx), by)

    train(model, bx, by, learning_rate=learning_rate)

    if i % log_steps == 0:
        print('Epoch {:4d} Step {:2d} Loss {:6.4f}'.format(int(i / steps_per_epoch), i, loss_val))

print('Final Parameters: ', model.w.numpy(), model.b.numpy())

X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training Examples', 'Linear Regression'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['Weight [W]', 'Bias Unit [B]'], fontsize=15)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Value', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()
