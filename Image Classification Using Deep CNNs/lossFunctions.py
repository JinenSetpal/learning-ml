import tensorflow as tf

# Binary Crossentropy
bce_probas = tf.keras.losses.BinaryCrossentropy(from_logits=False)
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

logits = tf.constant([0.8])
probas = tf.keras.activations.sigmoid(logits)

tf.print('BCE (w Probas): {:.4f}'.format(
    bce_probas(y_true=[1], y_pred=probas)),
        'w Logits: {:.4f}'.format(
            bce_logits(y_true=[1], y_pred=logits)))

# Categorical Crossentropy
cce_probas = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
cce_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

logits = tf.constant([[1.5, 0.8, 2.1]])
probas = tf.keras.activations.softmax(logits)

tf.print('CCE (w Probas): {:.4f}'.format(
    cce_probas(y_true=[[0, 0, 1]], y_pred=probas)),
        'w Logits: {:.4f}'.format(
            cce_logits(y_true=[[0, 0, 1]], y_pred=logits)))

# Sparse Categorical Crossentropy
sp_cce_probas = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
sp_cce_logits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

tf.print('Sparse CCE (w Probas): {:.4f}'.format(
    sp_cce_probas(y_true=[2], y_pred=probas)),
        'w Logits: {:.4f}'.format(
            sp_cce_logits(y_true=[2], y_pred=logits)))
