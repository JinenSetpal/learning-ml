import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


def create_samples(g_model, input_z):
    g_output = g_model(input_z, training=False)
    images = tf.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0


batch_size = 128
image_size = (28, 28)
z_size = 20
mode_z = 'uniform' # options: 'uniform' and 'normal'


if mode_z == 'uniform':
    fixed_z = tf.random.uniform(
        shape=(batch_size, z_size),
        minval=-1, maxval=1)
elif mode_z == 'normal':
    fixed_z = tf.random.normal(
        shape=(batch_size, z_size))

disc_model = tf.keras.models.load_model('DCGAN/disc_model')
gen_model = tf.keras.models.load_model('DCGAN/gen_model')

g_optimizer = pickle.load(open('DCGAN/g_optimizer.pkl', 'rb'))
d_optimizer = pickle.load(open('DCGAN/d_optimizer.pkl', 'rb'))

all_losses = []
# noinspection PyUnboundLocalVariable
epoch_samples = [create_samples(gen_model, fixed_z).numpy() for i in range(5)]
print(epoch_samples)

fig = plt.figure(figsize=(10, 14))
for i in range(len(epoch_samples) // 5):
    for j in range(5):
        ax = fig.add_subplot(6, 5, i * 5 + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        image = epoch_samples[i - 1][j]
        ax.imshow(image, cmap='gray_r')

plt.show()
