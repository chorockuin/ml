from pickletools import optimize
import tensorflow as tf
import matplotlib.pyplot as plt

class LinearRegression(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(1.0)
        self.b = tf.Variable(0.0)

    def call(self, x):
        return self.w * x + self.b

x = tf.random.normal([1000])
noise = tf.random.normal([1000])
w = 10.0
b = 3.0
y = w * x + b + noise

model = LinearRegression()

model.compile(optimizer= tf.keras.optimizers.SGD(0.1),
              loss=tf.keras.losses.MeanSquaredError())

model.fit(x, y, epochs=20, batch_size=1000)

plt.scatter(x, y)
plt.plot(x, model(x), 'r-')
plt.show()
