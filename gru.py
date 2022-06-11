import tensorflow as tf
import numpy as np

num_words = 2000
maxlen = 100

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words, maxlen=maxlen)
print(x_train)

def preprocess(x):
    for i in range(x.shape[0]): # sentence num
        x[i] = np.array([0 for _ in range(maxlen - len(x[i]))] + x[i])
    print(x)
    x = np.stack(x, axis=0)
    print(x.shape)
    return x

x_train = preprocess(x_train)
x_test = preprocess(x_test)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, 32, input_length=maxlen),
    tf.keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, 'sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.3)

results = model.evaluate(x_test, y_test)
print(f'result loss: {results[0]}, result accuracy: {results[1]}')