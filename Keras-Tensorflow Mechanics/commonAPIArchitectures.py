import tensorflow as tf

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.build(input_shape=(None, 4))
model.summary()

for v in model.variables:
    print('{:20s}'.format(v.name), v.trainable, v.shape)

# Timeline: Initialize --> Add Layers --> Configure Peripherals --> Fit training data --> Predict :-]
# Initialize the model
model = tf.keras.Sequential()
# Add layers to the NN
model.add(tf.keras.layers.Dense(units=16,
                                activation=tf.keras.activations.sigmoid,
                                kernel_initializer=tf.keras.initializers.glorot_normal(),
                                bias_constraint=tf.keras.initializers.Constant(20)))
model.add(tf.keras.layers.Dense(units=32,
                                activation=tf.keras.activations.sigmoid,
                                kernel_regularizer=tf.keras.regularizers.l1))
# Compile the model with peripherals
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              # learning_rate is a hyperparameter that has to be converted
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.Accuracy(),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(), ])
# Now the model is ready to be fitted and will react accordingly to data
