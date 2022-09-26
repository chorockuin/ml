import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis] * 2.0 - 1.0
x_test = x_test[..., tf.newaxis] * 2.0 - 1.0

data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = data.take(len(data) * 70 // 100).shuffle(1024, reshuffle_each_iteration=True).batch(32)
valid_data = data.skip(len(data) * 70 // 100).batch(128)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

model = Sequential([
    Conv2D(6, (5, 5), padding='valid', activation='tanh'),
    AveragePooling2D(),
    Conv2D(16, (5, 5), padding='valid', activation='tanh'),
    AveragePooling2D(),
    Flatten(),
    Dense(120, 'tanh'),
    Dense(84, 'tanh'),
    Dense(10, 'softmax')
])

model.compile('sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=valid_data)

results = model.evaluate(test_data)
print(f'reslut loss:{results[0]}, result accuracy:{results[1]}')
