import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = (x_train / 255.0) * 2.0 - 1.0
x_test = (x_test / 255.0) * 2.0 - 1.0

data = tf.data.Dataset.from_tensor_slices((x_train, x_train))

train_data = data.take(len(data) * 70 // 100).batch(32)
valid_data = data.skip(len(data) * 70 // 100).batch(128)
test_data = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(128)

class Encoder(tf.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = tf.keras.layers.Dense(1024, 'tanh')
        self.enc2 = tf.keras.layers.Dense(128, 'tanh') # embedding vector

    def __call__(self, x):
        x = self.enc1(x)
        return self.enc2(x)

class Decoder(tf.Module):
    def __init__(self):
        super().__init__()
        self.dec1 = tf.keras.layers.Dense(1024, 'tanh')
        self.dec2 = tf.keras.layers.Dense(784, 'tanh') # embedding vector

    def __call__(self, x):
        x = self.dec1(x)
        return self.dec2(x)

class DenoiseAutoEncoder(tf.keras.Model):
    def __init__(self, noise_std=0.2):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.noise_adder = tf.keras.layers.GaussianNoise(noise_std)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.reshape = tf.keras.layers.Reshape((28, 28))

    def call(self, x):
        x = self.flatten(x)
        x = self.noise_adder(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.reshape(x)

model = DenoiseAutoEncoder()

model.compile('adam', 'mse')

model.fit(train_data, epochs=2, validation_data=valid_data)

print(x_test[0].shape)
x_test0 = x_test[0][np.newaxis, ...] + np.random.normal(scale=0.2, size=(1, 28, 28))
print(x_test0.shape)
plt.imshow((x_test0[0] + 1.0) / 2.0, cmap=plt.cm.binary_r)
plt.show()

test_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    model.encoder,
    model.decoder,
    tf.keras.layers.Reshape((28, 28))
])

y = test_model(x_test0)[0]

plt.imshow((y + 1.0) / 2.0, cmap=plt.cm.binary_r) # convert -1.0 ~ 1.0 to 0.0 ~ 1.0
plt.show()
