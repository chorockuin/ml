from multiprocessing.dummy import active_children
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x, y), _ = mnist.load_data()
print(x.shape)
x = x / 255.0

data = tf.data.Dataset.from_tensor_slices((x, y))
data_num = len(data)

train_data_num = data_num * 35 // 100
valid_data_num = data_num * 15 // 100
test_data_num = data_num - (train_data_num + valid_data_num)
print(data_num, train_data_num, valid_data_num, test_data_num)

train_data = data.take(train_data_num).shuffle(1024, reshuffle_each_iteration=True).batch(32)
valid_data = data.skip(train_data_num).take(valid_data_num).batch(128)
test_data = data.skip(train_data_num + valid_data_num).batch(128)
print(len(train_data), len(valid_data), len(test_data))

class Model(tf.keras.Model):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='tanh')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense3 = tf.keras.layers.Dense(128, activation='tanh')
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.dense4 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        return self.dense4(x)

hists = []
for _ in range(1):
    model = Model()
    model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    hist = model.fit(train_data, epochs=50, validation_data=valid_data)
    hists.append(hist)

hists_dropout = []
for _ in range(1):
    model_dropout = Model(dropout_rate=0.5)
    model_dropout.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy())
    hist_dropout = model_dropout.fit(train_data, epochs=50, validation_data=valid_data)
    hists_dropout.append(hist_dropout)

iters = len(hists)
epochs = len(hist.epoch)
train_loss = np.zeros((epochs,))
valid_loss = np.zeros((epochs,))

for h in hists:
    train_loss += np.array(h.history['loss']) / iters
    valid_loss += np.array(h.history['val_loss']) / iters

train_dropout_loss = np.zeros((epochs,))
valid_dropout_loss = np.zeros((epochs,))
for h in hists_dropout:
    train_dropout_loss += np.array(h.history['loss']) / iters
    valid_dropout_loss += np.array(h.history['val_loss']) / iters

plt.plot(train_loss, 'r-', label='dropout=0.0')
plt.plot(valid_loss, 'r--')
plt.plot(train_dropout_loss, 'b-', label='dropout=0.5')
plt.plot(valid_dropout_loss, 'b--')
plt.legend()
plt.show()

# model = Model(0.5)
# model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')
# model.fit(train_data, epochs=16, validation_data=valid_data)
