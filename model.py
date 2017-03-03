import csv
import os
import cv2
import numpy as np

lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

imgs = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = os.path.split(os.path.realpath(__file__))[0] + '/IMG/' + filename
    img = cv2.imread(current_path)
    imgs.append(img)
    measurement = float(line[3])
    measurements.append(measurement)


X_train = np.array(imgs)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')