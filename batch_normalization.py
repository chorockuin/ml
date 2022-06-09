import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x, y), _ = mnist.load_data()
x = x / 255.0

data = tf.data.Dataset.from_tensor_slices((x, y))
data_num = len(data)

print(type(x), x.shape)
print(type(data), data_num)

train_data_num = data_num * 35 // 100
valid_data_num = data_num * 15 // 100

train_data = data.take(train_data_num).shuffle(1024).batch(32)
valid_data = data.skip(train_data_num).take(valid_data_num).batch(64)
test_data = data.skip(train_data_num + valid_data_num).batch(128)

class DenseBnRelu(tf.keras.layers.Layer):
    def __init__(self, feature_size):
        super().__init__()
        self.dense = tf.keras.layers.Dense(feature_size)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1) # feature_size x batch_size(axis=-1), feature_size x height x width x channel_size(axis=-1)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.dense(x)
        x = self.bn(x, training=training)
        return self.relu(x)

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = DenseBnRelu(32)
        self.layer2 = DenseBnRelu(64)
        self.layer3 = DenseBnRelu(128)
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        return self.dense(x)

model = Model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=valid_data)

results = model.evaluate(test_data)
print(f'reslut loss:{results[0]}, result accuracy:{results[1]}')
