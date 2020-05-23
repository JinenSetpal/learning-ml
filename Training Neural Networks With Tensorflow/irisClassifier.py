import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


iris, iris_info = tfds.load('iris', with_info=True)
print(iris_info)

tf.random.set_seed(1)
ds_orig = iris['train']

ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)
X_train = ds_orig.take(100)
X_test = ds_orig.skip(100)

ds_train = X_train.map(
    lambda x: [x['features'], x['label']])
ds_test = X_test.map(
    lambda x: [x['features'], x['label']])

iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='sigmoid',
                          name='fc1', input_shape=(4, )),
    tf.keras.layers.Dense(3, activation='softmax',
                          name='fc2')
])
print(iris_model.summary())

iris_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)

ds_train = ds_train.shuffle(buffer_size=100).repeat(count=None).batch(batch_size).prefetch(buffer_size=1000)

history = iris_model.fit(ds_train, epochs=num_epochs,
                         steps_per_epoch=steps_per_epoch,
                         verbose=1)

hist = history.history

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Training Loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training Accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

results = iris_model.evaluate(ds_test.batch(50), verbose=1)
print('Test loss: {:.4f} Test Acc: {:.4f}'.format(*results))

iris_model.save('iris-classifier.h5',
                overwrite=True,
                include_optimizer=True,
                save_format='h5')

iris_model = tf.keras.models.load_model('iris-classifier.h5')
print(iris_model.summary())
results = iris_model.evaluate(ds_test.batch(33), verbose=1)
print('Test loss: {:.4f} Test Acc: {:.4f}'.format(*results))
