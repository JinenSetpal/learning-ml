import tensorflow as tf

tf.random.set_seed(1)
rnn_layer = tf.keras.layers.SimpleRNN(
    units=2, use_bias=True,
    return_sequences=True)
rnn_layer.build(input_shape=(None, None, 5))
w_xh, w_oo, b_h = rnn_layer.weights
print('w_xh shape:', w_xh.shape)
print('w_oo shape:', w_oo.shape)
print('b_h shape:', b_h.shape)

x_seq = tf.convert_to_tensor([[1.0] * 5, [2.0] * 5, [3.0] * 5], dtype=tf.float32)
output = rnn_layer(tf.reshape(x_seq, shape=(1, 3, 5)))

out_man = []
for t in range(len(x_seq)):
    xt = tf.reshape(x_seq[t], (1, 5))
    ht = tf.matmul(xt, w_xh) + b_h
    print('Time step: {} ->'.format(t))
    print('\tInput\t\t\t:', xt.numpy())
    print('\tHidden\t\t\t:', ht.numpy())

    if t > 0:
        prev_o = out_man[t - 1]
    else:
        prev_o = tf.zeros(shape=ht.shape)
    ot = ht + tf.matmul(prev_o, w_oo)
    ot = tf.math.tanh(ot)

    out_man.append(ot)
    print('\tOut (manual)\t:', ot.numpy())
    print('\tSimpleRNN output:'.format(t), output[0][t].numpy())

# Example RNN using Tensorflow
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=32),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1)])
model.summary()
