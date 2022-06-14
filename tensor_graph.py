import tensorflow as tf
import numpy as np

# %load_ext tensorboard

# logdir = 'mylog'
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph=True)

@tf.function
def my_func(x, w, b):
    return x * w + b

x = tf.Variable(tf.random.uniform((3, 3)), name='intput')
w = tf.random.uniform((3, 3))
b = tf.random.uniform((3, 3))

y = my_func(x, w, b)
print(y)

# with writer.as_default():
#     tf.summary.trace_export(name='my_func_trace', step=0, profiler_outdir=logdir)

# %tensorboard --logdir mylog
