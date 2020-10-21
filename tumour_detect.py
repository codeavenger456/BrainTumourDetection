import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt
import cv2

nrows = 400
ncolumns = 300
channels = 1

def read_and_resize_image(list_of_images):
    X = list()
    y = list()
    for image in list_of_images:
        im = cv2.imread(image, 0)
        height = im.shape[0]
        if height > nrows:
            X.append(cv2.resize(im, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        else:
            X.append(cv2.resize(im, (nrows, ncolumns), interpolation=cv2.INTER_AREA))
        if 'yes' in image:
            y.append(1)
        else:
            y.append(0)
    return X, y

proportion = 0.8
images = list()
entries = os.listdir('.')
for i in entries:
    if "dataset" in i:
        for j in os.listdir(i):
            dataset = os.listdir(i + '/' + j + '/')
            image = [i + '/' + j + '/' + k for k in dataset]
            images.extend(image)

images_train = images[:int(len(images) * proportion)]
images_test = images[int(len(images) * proportion):]
X, y = read_and_resize_image(images_train)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)
X_train = np.expand_dims(X_train, axis=0)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(nrows, ncolumns, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(), metrics=[tf.keras.metrics.FalsePositives(name="falsePositive")])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
