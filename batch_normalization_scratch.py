import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x, y), _ = mnist.load_data()  # we use only train set
x = x / 255.0

data = tf.data.Dataset.from_tensor_slices((x, y))

train_data = data.take(len(data) * 35 // 100).shuffle(1024, reshuffle_each_iteration=True).batch(32)
valid_data = data.skip(len(data) * 35 // 100).take(len(data) * 15 // 100).batch(32)
test_data = data.skip(len(data) * 50 // 100)

class DenseBnRelu(tf.Module):
    def __init__(self, feature_size):
        self.dense = tf.keras.layers.Dense(feature_size)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
    
    def __call__(self, x, training=False):
        return self.relu(self.bn(self.dense(x), training))

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = DenseBnRelu(32)
        self.layer2 = DenseBnRelu(64)
        self.layer3 = DenseBnRelu(128)
        self.dense4 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.layer1(x, training)
        x = self.layer2(x, training)
        x = self.layer3(x, training)
        return self.dense4(x)

def loss_func(y, y_pred):
    return tf.losses.sparse_categorical_crossentropy(y, y_pred)

@tf.function
def train_step(model, x, y, optimizer, train_loss, train_accuracy):
        with tf.GradientTape() as t:
            y_pred = model(x, training=True)
            loss = loss_func(y, y_pred)
        grads = t.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss(loss)
        train_accuracy(y, y_pred)

@tf.function
def valid_step(model, x, y, valid_loss, valid_accuracy):
    y_pred = model(x, training=False)
    loss = loss_func(y, y_pred)

    valid_loss(loss)
    valid_accuracy(y, y_pred)

model = Model()    
optimizer = tf.optimizers.SGD(0.01)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

for epoch in range(10):
    for x, y in train_data:
        train_step(model, x, y, optimizer, train_loss, train_accuracy)
    
    for x, y in valid_data:
        valid_step(model, x, y, valid_loss, valid_accuracy)
    
    print(f'Epoch{epoch + 1} : train_loss: {train_loss.result()}, train_acc: {train_accuracy.result()}, valid_loss: {valid_loss.result()}, valid_acc: {valid_accuracy.result()}')

    train_loss.reset_state()
    train_accuracy.reset_state()
    valid_loss.reset_state()
    valid_accuracy.reset_state()

for x, y in test_data.batch(32):
    y_pred = model(x)
    test_accuracy(y, y_pred)

print(f'result accuracy: {test_accuracy.result()}')
test_accuracy.reset_state()

model = Model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=valid_data)

for x, y in test_data.batch(32):
    y_pred = model(x)
    test_accuracy(y, y_pred)
print(f'result accuracy: {test_accuracy.result()}')
test_accuracy.reset_state()
