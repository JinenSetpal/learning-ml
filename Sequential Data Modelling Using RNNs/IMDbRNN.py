import pandas as pd
import tensorflow as tf
from collections import Counter
import tensorflow_datasets as tfds


def encode(text_tensor, label):
    text = text_tensor.numpy()[0]
    encoded_text = encoder.encode(text)
    return encoded_text, label


def encode_map(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))


df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.columns = ['review', 'sentiment']

target = df.pop('sentiment')
ds_raw = tf.data.Dataset.from_tensor_slices((df.values, target.values))

tf.random.set_seed(1)
ds_raw = ds_raw.shuffle(50000, reshuffle_each_iteration=False)

ds_raw_test = ds_raw.take(25000)
ds_raw_train_valid = ds_raw.skip(25000)
ds_raw_train = ds_raw_train_valid.take(20000)
ds_raw_valid = ds_raw_train_valid.skip(20000)

tokenizer = tfds.features.text.Tokenizer()
token_counts = Counter()

for example in ds_raw_train:
    tokens = tokenizer.tokenize(example[0].numpy()[0])
    token_counts.update(tokens)

encoder = tfds.features.text.TokenTextEncoder(token_counts)

train_data = ds_raw_train.map(encode_map).padded_batch(32, padded_shapes=([-1], []))
valid_data = ds_raw_valid.map(encode_map).padded_batch(32, padded_shapes=([-1], []))
test_data = ds_raw_test.map(encode_map).padded_batch(32, padded_shapes=([-1], []))

embedding_dim = 20
vocab_size = len(token_counts) + 2

tf.random.set_seed(1)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed-layer'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, name='lstm-layer'), name='bsdir-lstm'),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.Accuracy()])

history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=10)
test_results = model.evaluate(test_data)
print('Test Acc.: {:2f}%'.format(test_results[1] * 100))
