import tensorflow as tf
import numpy as np

np.set_printoptions(precision=3)

a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]

# Creating Tensors
t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a, t_b, sep='\n')

# Casting Tensors
t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)

# Transposing a Tensor
t = tf.random.uniform(shape=(3, 5))
t_tr = tf.transpose(t)
print(t.shape, '-->', t_tr.shape)

# Reshaping a Tensor
t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t.shape, '-->', t_reshape.shape)

# Removing Unnecessary Dimensions
t = tf.zeros((1, 2, 1, 4, 1))
t_sqz = tf.squeeze(t, axis=(2, 4))
print(t.shape, '-->', t_sqz.shape)

# Initialization
tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5, 2),
                       minval=-1.0, maxval=1.0)
t2 = tf.random.normal(shape=(5, 2),
                      mean=0.0, stddev=1.0)

# Element-Wise Product
t3 = tf.multiply(t1, t2).numpy()
print(t3)

# Compute Mean
t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)

# Matrix Multiplication
t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
print(t5.numpy())
t6 = tf.linalg.matmul(t1, t2, transpose_a=True)
print(t6.numpy())

# Initialization
tf.random.set_seed(1)
t = tf.random.uniform((6, ))
print(t.numpy())

# Splitting Tensors
t_splits = tf.split(t, num_or_size_splits=3)
[print(item.numpy()) for item in t_splits]

# Uneven Tensor Split
tf.random.set_seed(1)
t = tf.random.uniform((5, ))
t_splits = tf.split(t, num_or_size_splits=[3, 2])
[print(item.numpy()) for item in t_splits]

# Concatenating Tensors
a = tf.zeros((3, ))
b = tf.ones((3, ))
c = tf.concat([a, b], axis=0)
print(c.numpy())

# Stacking Tensors
a = tf.zeros((3, ))
b = tf.ones((3, ))
c = tf.stack([a, b], axis=1)
print(c.numpy())

# Creating a Dataset from Tensor Slices
a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)

for item in ds:
    print(item)

ds_batch = ds.batch(3)
for i, elem in enumerate(ds_batch, 1):
    print('batch {}:'.format(i), elem.numpy())

# Initialization
tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=tf.float32)
t_y = tf.range(4)

ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)

# Combining two Tensorflow Datasets
ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
for example in ds_joint:
    print(' x:', example[0].numpy(),
          ' y:', example[1].numpy())

# Creating a Dataset out of multiple Tensor Slices
ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))
for example in ds_joint:
    print(' x:', example[0].numpy(),
          ' y:', example[1].numpy())

ds_trans = ds_joint.map(lambda x, y: (x * 2 - 1.0, y))
for element in ds_trans:
    print(' x:', element[0].numpy(),
          ' y:', element[1].numpy())

# Shuffling a Tensorflow Dataset
tf.random.set_seed(1)
ds = ds_joint.shuffle(buffer_size=len(t_x))
for element in ds:
    print(' x:', element[0].numpy(),
          ' y:', element[1].numpy())


ds = ds_joint.batch(batch_size=3,
                    drop_remainder=False)
batch_x, batch_y = next(iter(ds))
print('Batch-x:\n', batch_x.numpy())
print('Batch-y:\n', batch_y.numpy())

# Shuffle -> Batch -> Repeat [Best]
ds = ds_joint.shuffle(len(t_x)).batch(3).repeat(2)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# Batch -> Shuffle -> Repeat
ds = ds_joint.batch(3).shuffle(len(t_x)).repeat(2)
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())

# Batch -> Repeat -> Shuffle
ds = ds_joint.batch(3).repeat(2).shuffle(len(t_x))
for i, (batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())
