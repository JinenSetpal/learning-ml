import pandas as pd
import tensorflow as tf
from collections import Counter
import tensorflow_datasets as tfds


# noinspection PyShadowingNames
def preprocess_ds(ds_raw_train,
                  ds_raw_valid,
                  ds_raw_test,
                  max_seq_length=None,
                  batch_size=32):
    def encode(text_tensor, label):
        text = text_tensor.numpy()[0]
        encoded_text = encoder.encode(text)
        return encoded_text, label

    def encode_map(text, label):
        return tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

    tokenizer = tfds.features.text.Tokenizer()
    token_counts = Counter()

    for example in ds_raw_train:
        tokens = tokenizer.tokenize(example[0].numpy()[0])
        token_counts.update(tokens)

    encoder = tfds.features.text.TokenTextEncoder(token_counts)

    train_data = ds_raw_train.map(encode_map).padded_batch(32, padded_shapes=([-1], []))
    valid_data = ds_raw_valid.map(encode_map).padded_batch(32, padded_shapes=([-1], []))
    test_data = ds_raw_test.map(encode_map).padded_batch(32, padded_shapes=([-1], []))

    return train_data, valid_data, test_data, len(token_counts)


# noinspection PyShadowingNames,PyGlobalUndefined
def build_rnn_model(embedding_dim, vocab_size,
                    recurrent_type='SimpleRNN',
                    n_recurrent_units=64,
                    n_recurrent_layers=1,
                    bidirectional=True):
    global recurrent_layer
    tf.random.set_seed(1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embed-layer'))

    for i in range(n_recurrent_layers):
        return_sequences = (i < n_recurrent_layers - 1)
        if recurrent_type == 'SimpleRNN':
            recurrent_layer = tf.keras.layers.SimpleRNN(units=n_recurrent_units, return_sequences=return_sequences, name='simplernn-layer-{}'.format(i))
        elif recurrent_type == 'LSTM':
            recurrent_layer = tf.keras.layers.LSTM(units=n_recurrent_units, return_sequences=return_sequences, name='lstm-layer={}'.format(i))
        elif recurrent_type == 'GRU':
            recurrent_layer = tf.keras.layers.GRU(units=n_recurrent_units, return_sequences=return_sequences, name='gru-layers-{}'.format(i))
        if bidirectional:
            recurrent_layer = tf.keras.layers.Bidirectional(recurrent_layer, name='bidir-{}'.format(recurrent_layer.name))
        model.add(recurrent_layer)
    model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
    return model


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

batch_size = 32
embedding_dim = 20
max_seq_length = 100

train_data, valid_data, test_data, tokens = preprocess_ds(ds_raw_train, ds_raw_valid, ds_raw_test, max_seq_length=max_seq_length, batch_size=batch_size)
vocab_size = tokens + 2

model = build_rnn_model(embedding_dim, vocab_size)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.Accuracy()])
model.summary()
history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=10)
model.save('imdb-functions-analysis')

test_results = model.evaluate(test_data)
print('Test Acc.: {:2f}%'.format(test_results[1] * 100))
