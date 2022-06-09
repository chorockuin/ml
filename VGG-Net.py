import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

vgg16 = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
vgg16.summary()

img = cv2.imread('data/cat.jpg') # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB
img = cv2.resize(img, (224, 224))
plt.imshow(img)

img = img / 255.0 # 0.0 ~ 1.0
img = tf.constant(img)[tf.newaxis, ...] # NHWC

y_pred = vgg16.predict(img)
print(y_pred)
print(tf.keras.applications.vgg16.decode_predictions(y_pred))
