import tensorflow as tf
import numpy as np

@tf.function
def my_func(x, w, b):
    return x * w + b

x = tf.Variable(tf.random.uniform((3, 3)), name='intput')
w = tf.random.uniform((3, 3))
b = tf.random.uniform((3, 3))

y = my_func(x, w, b)
print(y)
