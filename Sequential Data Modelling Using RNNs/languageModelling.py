import tensorflow as tf
import numpy as np
import os


def split_input_target(chunk):
    input_seq = chunk[:-1]
    target_seq = chunk[1:]
    return input_seq, target_seq


# noinspection PyShadowingNames
def build_model(vocab_size, embedding_dim, rnn_units):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True),
        tf.keras.layers.Dense(vocab_size)])


# noinspection PyShadowingNames
def sample(model, starting_str,
           len_generated_text=500,
           max_input_length=40,
           scale_factor=1.0):
    encoded_input = tf.reshape([char_to_int[s] for s in starting_str], (1, -1))

    generated_str = starting_str

    model.reset_states()
    for i in range(len_generated_text):
        logits = tf.squeeze(model(encoded_input), 0)
        new_char_index = tf.squeeze(tf.random.categorical(logits * scale_factor, num_samples=1))[-1].numpy()
        generated_str += str(char_array[new_char_index])
        encoded_input = tf.concat([encoded_input, tf.expand_dims([new_char_index], 0)], axis=1)[:, -max_input_length:]

    return generated_str


with open('training_book.txt', 'r') as file:
    text = file.read()

start_index = text.find('THE MYSTERIOUS ISLAND')
end_index = text.find('End of Project Gutenberg')
text = text[start_index:end_index]

char_set = set(text)
print('Total Length:', len(text))
print('Unique Characters:', len(char_set))

chars_sorted = sorted(char_set)
char_to_int = {ch: i for i, ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array([char_to_int[ch] for ch in text], dtype=np.int32)

print('Text Encoded Shape:', text_encoded.shape)
print(text[:15], '== Encoding ==>', text_encoded[:15])
print(text_encoded[15:21], '== Reversed ==>', text[15:21])

batch = 64
buffer = 10000
ds = tf.data.Dataset.from_tensor_slices(text_encoded).batch(41, drop_remainder=True).map(split_input_target).shuffle(buffer).batch(batch)

charset_size = len(char_array)
embedding_dim = 256
rnn_units = 512

tag = 1
if os.path.exists('passage-generator') and tag == 1:
    model = tf.keras.models.load_model('passage-generator')
else:
    tf.random.set_seed(1)
    model = build_model(charset_size,
                        embedding_dim,
                        rnn_units)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(ds, epochs=20)
    model.save('passage-generator')

tf.random.set_seed(1)
logits = [[1.0, 1.0, 1.0]]
print('Probabilities:', tf.math.softmax(logits).numpy()[0])
samples = tf.random.categorical(logits=logits, num_samples=10)
tf.print(samples.numpy())

tf.random.set_seed(1)
logits = [[1.0, 1.0, 3.0]]
print('Probabilities:', tf.math.softmax(logits).numpy()[0])
samples = tf.random.categorical(logits=logits, num_samples=10)
tf.print(samples.numpy())

print(sample(model, 'The island'))
