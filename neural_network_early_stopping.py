from multiprocessing.dummy import active_children
import tensorflow as tf
import matplotlib.pyplot as plt

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
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        return self.dense4(self.dense3(self.dense2(self.dense1(self.flatten(x)))))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, restore_best_weights=True)

model = Model()

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')
model.fit(train_data, epochs=100, validation_data=valid_data, callbacks=[callback])

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.show()
