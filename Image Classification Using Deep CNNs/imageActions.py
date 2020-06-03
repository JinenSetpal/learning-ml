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


def count_items(ds):
    n = 0
    for _ in ds:
        n += 1
    return n


celeba_bldr = tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)

celeba_train, celeba_valid, celeba_test = celeba['train'], celeba['validation'], celeba['test']

celeba_train = celeba_train.take(160000)
celeba_valid = celeba_valid.take(1000)

examples = []
for example in celeba_train.take(5):
    examples.append(example['image'])

fig = plt.figure(figsize=(16, 8.5))

ax = fig.add_subplot(2, 5, 1)
ax.set_title('Crop to a \nbounding-box', size=15)
ax.imshow(examples[0])
ax = fig.add_subplot(2, 5, 6)
img_cropped = tf.image.crop_to_bounding_box(examples[0], 50, 20, 128, 128)
ax.imshow(img_cropped)

ax = fig.add_subplot(2, 5, 2)
ax.set_title('Flip (horizontal)', size=15)
ax.imshow(examples[1])
ax = fig.add_subplot(2, 5, 7)
img_flipped = tf.image.flip_left_right(examples[1])
ax.imshow(img_flipped)

ax = fig.add_subplot(2, 5, 3)
ax.set_title('Adjust Contrast', size=15)
ax.imshow(examples[2])
ax = fig.add_subplot(2, 5, 8)
img_adj_contrast = tf.image.adjust_contrast(examples[2], contrast_factor=2)
ax.imshow(img_adj_contrast)

ax = fig.add_subplot(2, 5, 4)
ax.set_title('Adjust Brightness')
ax.imshow(examples[3])
ax = fig.add_subplot(2, 5, 9)
img_adj_brightness = tf.image.adjust_brightness(examples[3], delta=0.3)
ax.imshow(img_adj_brightness)

ax = fig.add_subplot(2, 5, 5)
ax.set_title('Central crop \nand resize')
ax.imshow(examples[4])
ax = fig.add_subplot(2, 5, 10)
img_central_crop = tf.image.central_crop(examples[4], 0.7)
img_resized = tf.image.resize(img_central_crop, size=(218, 178))
ax.imshow(img_resized.numpy().astype('uint8'))
plt.show()

tf.random.set_seed(1)
ds = celeba_train.shuffle(1000, reshuffle_each_iteration=False)
ds = ds.take(2).repeat(5)
ds = ds.map(
    lambda x: preprocess(x, size=(178, 178), mode='train'))

fig = plt.figure(figsize=(15, 6))
for i, example in enumerate(ds):
    ax = fig.add_subplot(2, 5, i // 2 + (i % 2) * 5 + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
plt.show()
