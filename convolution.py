import tensorflow as tf

x = tf.random.normal((32, 128, 128, 3))

conv_layer = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(3, 3),
                                    strides=(1,1),
                                    padding='same',
                                    activation='relu')

y = conv_layer(x)
print(y.shape)