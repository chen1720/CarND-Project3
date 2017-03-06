import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


samples = []

def add_samples(csv_filepath, samples):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            samples.append(sample)
        return samples

car_imgs = []
steering_angles = []

samples = add_samples('driving_log.csv', samples)
samples = add_samples('driving_log_opposite.csv', samples)

print("Samples: ",len(samples))

def generator(samples, batch_size=32):

    for line in samples:
        current_path = os.path.split(os.path.realpath(__file__))[0] + '/IMG/' + line[0].split('\\')[-1]
        img_center = cv2.imread(current_path)

        current_path = os.path.split(os.path.realpath(__file__))[0] + '/IMG/' + line[1].split('\\')[-1]
        img_left = cv2.imread(current_path)

        current_path = os.path.split(os.path.realpath(__file__))[0] + '/IMG/' + line[2].split('\\')[-1]
        img_right = cv2.imread(current_path)

        car_imgs.extend([img_center, img_left, img_right])

        steering_center = float(line[3])

        correction = 0.2
        steering_left =  steering_center + correction
        steering_right = steering_center - correction

        steering_angles.extend([steering_center, steering_left, steering_right])

    augmented_images, augmented_measurements = [], []

    for img, measurement in zip(car_imgs, steering_angles):
        augmented_images.append(img)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(img,1))
        augmented_measurements.append(measurement*-1.0)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    yield shuffle(X_train, y_train)

X_train, y_train = generator(samples)
print("Training samples:", len(X_train))

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10 , verbose=1)

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
model.save('model.h5')