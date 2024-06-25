# backend/utils/data_generator.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, file_list, labels, batch_size=32, dim=(256, 256), n_channels=3, shuffle=True):
        self.file_list = file_list
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        file_list_temp = [self.file_list[k] for k in indexes]
        X, y = self.__data_generation(file_list_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_list_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_age = np.empty((self.batch_size), dtype=int)
        y_gender = np.empty((self.batch_size), dtype=int)
        y_race = np.empty((self.batch_size, 5), dtype=int)  # Assuming 5 races

        for i, file_name in enumerate(file_list_temp):
            img = cv2.imread(file_name)
            img = cv2.resize(img, self.dim)
            img = img.astype('float32') / 255.0

            X[i,] = img
            age, gender, race = self.labels[file_name]
            y_age[i] = age
            y_gender[i] = gender
            y_race[i,] = tf.keras.utils.to_categorical(race, num_classes=5)

        return X, {'age_output': y_age, 'gender_output': y_gender, 'race_output': y_race}
