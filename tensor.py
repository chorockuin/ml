import tensorflow as tf
import numpy as np

print(tf.constant(3))
print(tf.constant(5, dtype=tf.float32))
print(tf.constant([2.0, 3.0, 4.0, 5.0]))
print(tf.constant([[1, 2], [3, 4]], dtype=tf.float16))
print(tf.constant(np.array(range(27)).reshape(3, 3, 3)))
print(tf.constant(range(27), shape=(3, 3, 3)))

a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
print(a.numpy())
print(b.numpy())

print(tf.add(a, b).numpy())
print(tf.subtract(a, b))
print(tf.multiply(a, b))
print(tf.divide(a, b))
print(tf.matmul(a, b))

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a @ b)

print(tf.zeros((3, 3, 2)))
print(tf.ones((1, 4, 2)))
print(tf.ones_like(a))
print(tf.eye(5))
print(tf.eye(5, 2))
print(tf.eye(2, 5))

a = tf.ones((2, 3, 4, 5))
print(a.ndim, a.dtype, a.shape, a.shape[0], a.shape[1], a.shape[2])
print(tf.size(a).numpy())

a = tf.constant(range(27), shape=(3, 3, 3))
print(a[0][0][0].numpy())
print(a[0, 0, 0].numpy())
print(a[0, :, :].numpy())
print(a[:, 0, :].numpy())
print(a[:, :, 0].numpy())
print(a[0, 1:, :2].numpy())
print(a[..., :2].numpy())
print(a[..., :2, :2].numpy())
print(a[:, :, 0].numpy())
print(a[:, :, 0:1].numpy())
print(tf.reshape(a[:, :, 0], (3, 3, 1)).numpy())
print(tf.reshape(a[:, :, 0], (3, 3, -1)).numpy())
print(tf.reshape(a[:, :, 0], (3, -1, 1)).numpy())
print(tf.reshape(a[:, :, 0], a.shape[:-1] + [1]).numpy())

b = a[:, :, 0]
print(b[..., tf.newaxis].numpy())
print(b[tf.newaxis, ..., tf.newaxis].numpy())

var0 = tf.Variable(0)
var1 = tf.Variable(np.array([1, 2, 3]), dtype=tf.float16, name='var1')
print(var1)
print(var1[tf.argmax(var1)])
print(var1.assign([2, 3, 4]))
print(var1.assign(var1 + tf.constant([1,2, 3], dtype=tf.float16)))
print(var1.assign_add(tf.constant([1, 2, 3], dtype=tf.float16)))

a = tf.Variable(1, name='var')
b = tf.Variable(2, name='var')
print(a == b)
counter = tf.Variable(0, trainable=False)
print(counter)