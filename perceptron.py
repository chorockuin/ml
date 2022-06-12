import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

a = tf.Variable([[1, 2]])
b = tf.Variable([[3], [4]])
print(a * b)
print(b * a)
print(a @ b)
print(tf.transpose(b) * a)

x, y = make_classification(n_samples=100, 
                           n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           n_classes=2,
                           n_clusters_per_class=1,
                           class_sep=2.0)
y = 2 * y - 1

plt.scatter(x[:, 0], x[:, 1], c=y)

class Perceptron(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(tf.random.normal([2, 1]), dtype=tf.float32)
        self.b = tf.Variable(0.0, dtype=tf.float32)

    def __call__(self, x):
        return tf.reduce_sum(tf.transpose(self.w) * x, axis=1) + self.b
        
model = Perceptron()
print(model.w, model.b)
print(tf.sign(model(x)))

def loss_func(y, y_pred, z):
    return -(y - y_pred) * z

def train_step(model, x, y, learning_rate):
    for _x, _y in zip(x, y):
        with tf.GradientTape() as t:
            z = model(_x)
            y_pred = tf.sign(z)
            loss = loss_func(_y, y_pred, z)
        dw, db = t.gradient(loss, [model.w, model.b])
        model.w.assign_sub(learning_rate * dw)
        model.b.assign_sub(learning_rate * db)

train_step(model, x, y, learning_rate=0.1)

plt.scatter(x[:, 0], x[:, 1], c=tf.sign(model(x)))
plt.show()
