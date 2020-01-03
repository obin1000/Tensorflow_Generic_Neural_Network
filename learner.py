import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np


class Learner:
    EPOCHS = 10  # Number of times to see the dataset

    def __init__(self, labels, dataset):
        """
        Labels indexes should match the indexes in the dataset.
        :param labels:
        :param dataset: Data of the images [[image1], [image2]...]
        """

        self.model = Sequential()

        self.labels = labels
        self.dataset = dataset

        self.add_input_layer()
        self.add_hidden_layer()
        self.add_result_layer()

    def train(self):
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit(np.asarray(self.dataset), self.labels, epochs=self.EPOCHS)

    def add_input_layer(self):
        """
        Same size as input number of pixels in input data
        :return:
        """
        # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Flatten(input_shape=self.dataset[0].shape))

    def add_hidden_layer(self):
        """
        size is 15/20% of input size
        :return:
        """
        self.model.add(Dense(128, activation='relu'))
        # Make all results add up to 1
        self.model.add(Dense(10, activation='softmax'))

    def add_result_layer(self):
        pass

    def test(self, labels, data):
        test_loss, test_acc = self.model.evaluate(labels, data)
        print('Loss: {} \n Acc: {}'.format(test_loss, test_acc))
