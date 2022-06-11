import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape) # (60000, 28, 28) = (N, time-stamp, vector)

data = tf.data.Dataset.from_tensor_slices((x_train / 255.0, y_train))

train_data = data.take(len(data) * 70 // 100).shuffle(1024).batch(32)
valid_data = data.skip(len(data) * 70 // 100).batch(128)
test_data = tf.data.Dataset.from_tensor_slices((x_test / 255.0, y_test)).batch(128)

class VanillaRNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.rnn = tf.keras.layers.SimpleRNN(128)
        self.dense = tf.keras.layers.Dense(10, 'softmax')

    def call(self, x):
        x = self.rnn(x)
        return self.dense(x)

model = VanillaRNN()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=20, validation_data=valid_data)

results = model.evaluate(test_data)
print(f'result loss: {results[0]}, result accuracy: {results[1]}')