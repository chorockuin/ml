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

def loss_func(y, y_pred):
    return tf.losses.sparse_categorical_crossentropy(y, y_pred)

@tf.function
def train_step(model, x, y, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as t:
        y_pred = model(x)
        loss = loss_func(y, y_pred)
    grads = t.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_loss(loss)
    train_accuracy(y, y_pred)

@tf.function
def valid_step(model, x, y, valid_loss, valid_accuracy):
    y_pred = model(x)
    loss = loss_func(y, y_pred)
    valid_loss(loss)
    valid_accuracy(y, y_pred)

model = Model(0.5)

optimizer = tf.optimizers.SGD(0.01)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('valid_accuracy')

test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

for epoch in range(16):
    for x_batch, y_batch, in train_data:
        train_step(model, x_batch, y_batch, optimizer, train_loss, train_accuracy)

    for x_batch, y_batch in valid_data:
        valid_step(model, x_batch, y_batch, valid_loss, valid_accuracy)

    print(f'epoch{epoch+1}: train_loss:{train_loss.result()} train_accuracy:{train_accuracy.result()} valid_loss:{valid_loss.result()} valid_accuracy:{valid_accuracy.result()}')

    train_loss.reset_state()
    train_accuracy.reset_state()
    valid_loss.reset_state()
    valid_accuracy.reset_state()

for x_batch, y_batch in test_data:
    y_pred = model(x_batch)
    test_accuracy(y_batch, y_pred)

print(f'test_accuracy:{test_accuracy.result()}')