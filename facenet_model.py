import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_root = r'.\data\train'
print(data_root)

IMAGE_SHAPE = (299, 299)
batch_size = 8

image_generator = ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(
                                    str(data_root), 
                                    target_size=IMAGE_SHAPE,
                                    batch_size=batch_size)

for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break


# model load
hub_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/4"

input_shape = (299,299,3)
hub_layer = hub.KerasLayer(hub_url,
                           input_shape=input_shape, trainable=True)

model = Sequential()
model.add(hub_layer)
model.add(Dense(128, activation='relu'))

adam = Adam(lr=1e-3, decay=1e-6)
model.compile(optimizer=adam, loss=tfa.losses.TripletSemiHardLoss(), metrics=['accuracy'])

model.summary()

print(image_data.batch_size)
print(image_data.samples)

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
print(steps_per_epoch)

history = model.fit_generator(image_data, epochs=1,
                              steps_per_epoch=steps_per_epoch)
