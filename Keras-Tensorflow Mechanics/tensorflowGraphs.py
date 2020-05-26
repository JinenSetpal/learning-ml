import tensorflow as tf


# noinspection PyShadowingNames
@tf.function(input_signature=(tf.TensorSpec(shape=[None],
                                            dtype=tf.int32),
                              tf.TensorSpec(shape=[None],
                                            dtype=tf.int32),
                              tf.TensorSpec(shape=[None],
                                            dtype=tf.int32),))
def compute_z(a, b, c):
    r1 = tf.subtract(a, b)
    r2 = tf.multiply(2, r1)
    return tf.add(r2, c)


class myModule(tf.Module):
    def __init__(self):
        super().__init__()
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)),
                              trainable=True)
        self.w2 = tf.Variable(init(shape=(1, 2)),
                              trainable=False)


a = tf.constant(1, name='a')
b = tf.constant(2, name='b')
c = tf.constant(3, name='c')
z = 2 * (a - b) + c

tf.print('Result:', z)

# tf.print('Scalar Inputs:', compute_z(a, b, c))
tf.print('Rank 1 Inputs:', compute_z([a], [b], [c]))
# tf.print('Rank 2 Inputs:', compute_z([[a]], [[b]], [[c]]))

m = myModule()
print('All module variables:', [v.shape for v in m.variables])
print('Trainable Variables:', [v.shape for v in m.trainable_variables])

w = tf.Variable(1.0)
b = tf.Variable(0.5)
print(w.trainable, b.trainable)

x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])
with tf.GradientTape(persistent=True) as tape:
    # tape.watch(variable) <-- for variables where trainable is set to False
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

dloss_dw = tape.gradient(loss, w)
tf.print('dL/dw:', dloss_dw)

dloss_db = tape.gradient(loss, b)
tf.print('dL/dw:', dloss_db)

optimizer = tf.keras.optimizers.SGD()
optimizer.apply_gradients(zip([dloss_db, dloss_dw], [b, w]))

tf.print('Updated w:', w)
tf.print('Updated b:', b)
