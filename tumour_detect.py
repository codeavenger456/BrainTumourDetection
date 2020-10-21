import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt
import cv2

nrows = 400 #height of image
ncolumns = 300 #width of image
channels = 1 #grayscale

def read_and_resize_image(list_of_images):
    #convert list of path to pair of array of grayscale and array of reponses
    X = list()
    y = list()
    for image in list_of_images:
        im = cv2.imread(image, 0)
        height = im.shape[0]
        if height > nrows:
            #if original image height is bigger than we make it smaller
            X.append(cv2.resize(im, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        else:
            # else we make it bigger
            X.append(cv2.resize(im, (nrows, ncolumns), interpolation=cv2.INTER_AREA))
        if 'yes' in image:
            y.append(1)
        else:
            y.append(0)
    return X, y

proportion = 0.8 #80% of the dataset is training data
images = list() #path of images

#parse dataset
entries = os.listdir('.')
for i in entries:
    if "dataset" in i:
        for j in os.listdir(i):
            dataset = os.listdir(i + '/' + j + '/')
            image = [i + '/' + j + '/' + k for k in dataset]
            images.extend(image)

images_train = images[:int(len(images) * proportion)]
images_test = images[int(len(images) * proportion):]

train_X, train_y = read_and_resize_image(images_train)
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=2) #separate 20% of training data to validation set

X_train = np.expand_dims(X_train, axis=0)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

test_X, test_y = read_and_resize_image(images_test)

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
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val)) #ERROR occurs here due to shape conflict!!!!!

plt.plot(history.history["falsePositive"], label="falsePositive")
plt.xlabel('Epoch')
plt.ylabel('FalsePositive')
plt.show()