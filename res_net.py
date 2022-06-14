import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

resnet = tf.keras.applications.ResNet101V2(include_top=True, 
                                           weights='imagenet', 
                                           classes=1000,
                                           classifier_activation='softmax')
resnet.summary()

img = cv2.imread('data/cat.jpg') # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB
img = cv2.resize(img, (224, 224))
plt.imshow(img)

img = img / 255.0 # 0.0 ~ 1.0
img = tf.constant(img)[tf.newaxis, ...] # NHWC

print(tf.keras.applications.resnet_v2.decode_predictions(resnet.predict(img)))
