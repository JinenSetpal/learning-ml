import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os


def preprocess(item):
    image = item['image']
    label = item['label']
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, (-1,))
    return {'image-pixels': image}, label[..., tf.newaxis]


def train_input():
    dataset = tfds.load(name='mnist')
    mnist_train = dataset['train']
    return mnist_train.map(preprocess).shuffle(buffer_size).batch(batch_size).repeat()


def eval_input():
    dataset = tfds.load(name='mnist')
    mnist_test = dataset['test']
    return mnist_test.map(preprocess).batch(batch_size)


buffer_size = 10000
batch_size = 64
epochs = 20
steps = np.ceil(60000 / batch_size)

image_feature_column = tf.feature_column.numeric_column(key='image-pixels', shape=(28 * 28))

if os.path.exists('models/mnist-dnn'):
    dnn_classifer = tf.estimator.DNNClassifier(feature_columns=[image_feature_column],
                                               hidden_units=[32, 16],
                                               n_classes=10,
                                               warm_start_from='models/mnist-dnn',
                                               model_dir='models/mnist-dnn')

else:
    dnn_classifer = tf.estimator.DNNClassifier(feature_columns=[image_feature_column],
                                               hidden_units=[32, 16],
                                               n_classes=10,
                                               model_dir='models/mnist-dnn')

dnn_classifer.train(input_fn=train_input,
                    steps=epochs * steps)
eval_result = dnn_classifer.evaluate(input_fn=eval_input)
print(eval_result)
