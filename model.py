import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

imgs = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = os.path.split(os.path.realpath(__file__))[0] + '/IMG/' + filename
    img = cv2.imread(current_path)
    imgs.append(img)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
print(current_path)

for img, measurement in zip(imgs, measurements):
    augmented_images.append(img)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(img,1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# from keras.models import Sequential
# from keras.layers import Cropping2D
# from keras.layers.core import Flatten, Dense, Lambda
# from keras.layers.convolutional import Convolution2D
# from keras.layers.pooling import MaxPooling2D
#
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((50,20),(0,0))))
# model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
# model.add(Convolution2D(64, 3, 3, activation="relu"))
# model.add(Convolution2D(64, 3, 3, activation="relu"))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))
#
# model.compile(loss='mse', optimizer='adam')
#
# history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)
# #
# # ### print the keys contained in the history object
# # print(history_object.history.keys())
# #
# # ### plot the training and validation loss for each epoch
# # plt.plot(history_object.history['loss'])
# # plt.plot(history_object.history['val_loss'])
# # plt.title('model mean squared error loss')
# # plt.ylabel('mean squared error loss')
# # plt.xlabel('epoch')
# # plt.legend(['training set', 'validation set'], loc='upper right')
# # plt.show()
#
# model.save('model.h5')