import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=100, 
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_classes=2,
                           n_clusters_per_class=1,
                           class_sep=2.0)

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

for _x, _y in zip(x, y):
    print(_x, _x.shape, _y, _y.shape)

class NeuralNetwork(tf.Module):
    def __init__(self, n_input, n_hidden, n_output, **kwargs):
        super().__init__(**kwargs)
        self.w_h = tf.Variable(tf.random.normal([n_input, n_hidden], dtype=tf.float32))     # (2, 128)
        self.b_h = tf.Variable(tf.random.uniform([n_hidden, 1], dtype=tf.float32))          # (128, 1)
        self.w_o = tf.Variable(tf.random.normal([n_hidden, n_output], dtype=tf.float32))    # (128, 1)
        self.b_o = tf.Variable(tf.random.uniform([n_output, 1], dtype=tf.float32))          # (1, 1)

    def __call__(self, x):
        i = x[..., tf.newaxis]
        h = tf.sigmoid(tf.transpose(self.w_h) @ i + self.b_h)
        o = tf.sigmoid(tf.transpose(self.w_o) @ h + self.b_o)
        print(x.shape, i.shape, h.shape, o.shape)
        return o

model = NeuralNetwork(2, 128, 1)

def loss_func(y, y_pred):
    return -y * tf.math.log(y_pred) - (1 - y) * tf.math.log(1 - y_pred)

def train_step(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        loss = 0.0
        for _x, _y in zip(x, y):
            y_pred = model(_x)
            loss += loss_func(_y, y_pred) / x.shape[0]
    grads = t.gradient(loss, model.trainable_variables)
    for var, grad in zip(model.trainable_variables, grads):
        var.assign_sub(learning_rate * grad)
    return loss.numpy()

for epoch in range(10):
    loss = train_step(model, x, y, 0.1)
    print(f'epoch: {epoch+1} loss: {loss}')

plt.scatter(x[:, 0], x[:, 1], c=model(x))
# plt.scatter(x[:, 0], x[:, 1], c=model(x) > 0.5)
plt.show()
