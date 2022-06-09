from multiprocessing.dummy import active_children
import tensorflow as tf
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

model = Model(0.5)

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')
model.fit(train_data, epochs=16, validation_data=valid_data)
