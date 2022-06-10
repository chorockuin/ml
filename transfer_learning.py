import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# https://www.kaggle.com/yashvrdnjain/hotdognothotdog

data = tf.keras.preprocessing.image_dataset_from_directory(
    'data/hotdog-nothotdog/train/',
    image_size=(224, 224),
    batch_size=32)

train_data = data.take(len(data) * 70 // 100)
valid_data = data.skip(len(data) * 70 // 100)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'data/hotdog-nothotdog/test/',
    image_size=(224, 224),
    batch_size=128)
    
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet")

base_model.trainable = False

callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])

model.fit(train_data, epochs=100, validation_data=valid_data, callbacks=[callback])

result = model.evaluate(test_data)
print(f'test loss: {result[0]}, test accuracy: {result[1]}')
