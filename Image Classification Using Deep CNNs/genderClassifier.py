import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


# noinspection PyShadowingNames
def preprocess(example, size=(64, 64), mode='train'):
    image = example['image']
    label = example['attributes']['Male']
    if mode == 'train':
        image_cropped = tf.image.random_crop(image, size=(178, 178, 3))
        image_resized = tf.image.resize(image_cropped, size=size)
        image_flipped = tf.image.flip_left_right(image_resized)
        return image_flipped / 255.0, tf.cast(label, tf.int32)
    else:
        image_cropped = tf.image.crop_to_bounding_box(
            image, offset_height=20, offset_width=0,
            target_height=178, target_width=178)
        image_resized = tf.image.resize(
            image_cropped, size=size)
        return image_resized / 255.0, tf.cast(label, tf.int32)


batch = 32
buffer = 1000
image_size = (64, 64)
epochs = np.ceil(16000 / batch)

celeba_bldr = tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)

celeba_train, celeba_valid, celeba_test = celeba['train'], celeba['validation'], celeba['test']
celeba_train = celeba_train.take(160000)
celeba_valid = celeba_valid.take(1000)

ds_train = celeba_train.map(
    lambda x: preprocess(x, size=image_size, mode='train')).shuffle(buffer_size=buffer).repeat().batch(batch)
ds_valid = celeba_valid.map(
    lambda x: preprocess(x, size=image_size, mode='eval')).batch(batch)
ds_test = celeba_test.map(
    lambda x: preprocess(x, size=image_size, mode='eval')).batch(32)

if not os.path.exists('gender_classification'):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same', activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same', activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(
            128, (3, 3), padding='same', activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            256, (3, 3), padding='same', activation=tf.keras.activations.relu),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation=None)
    ])

    tf.random.set_seed(1)
    model.build(input_shape=(None, 64, 64, 3))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(ds_train, validation_data=ds_valid,
                        epochs=30,
                        steps_per_epoch=epochs)
    model.save('gender_classification')

    hist = history.history
    x_arr = np.arange(len(hist['loss'])) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist['loss'], '-o', label='Train Loss')
    ax.plot(x_arr, hist['val_loss'], '--<', label='Validation Loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist['accuracy'], '-o', label='Train Acc')
    ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation Acc')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    ax.legend(fontsize=15)
    plt.show()
else:
    model = tf.keras.models.load_model('gender_classification')

test_results = model.evaluate(ds_test)
print('Test Accruacy: {:.2f}%'.format(test_results[1] * 100))

ds = ds_test.unbatch().take(10)
pred_logits = model.predict(ds.batch(10))
probas = tf.sigmoid(pred_logits)
probas = probas.numpy().flatten() * 100

fig = plt.figure(figsize=(15, 7))
for i, example in enumerate(ds):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])

    if example[1].numpy() == 1:
        label = 'M'
    else:
        label = 'F'
    ax.text(
        0.5, -0.15, 'GT: {:s}\nPr(Male)={:.0f}%'
        ''.format(label, probas[i]),
        size=16,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)
plt.tight_layout()
plt.show()
