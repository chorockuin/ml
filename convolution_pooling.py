import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D

x = tf.random.normal((32, 128, 128, 3))

print(MaxPooling2D()(x).shape)
print(MaxPooling2D(pool_size=(2,2),
                   strides=(2,2),
                   padding='valid')(x).shape)
print(MaxPooling2D(pool_size=(2,2),
                   strides=(5,5),
                   padding='valid')(x).shape)
print(MaxPooling2D(pool_size=(5,5),
                   strides=(2,2),
                   padding='valid')(x).shape)
print(MaxPooling2D(pool_size=(5,5),
                   padding='valid')(x).shape)
print(MaxPooling2D(pool_size=(5,5),
                   padding='same')(x).shape)