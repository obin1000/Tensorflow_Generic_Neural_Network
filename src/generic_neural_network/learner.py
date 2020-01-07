import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np


class Learner:
    EPOCHS = 10  # Number of times to see the dataset
    LEARNING_RATE = 0.01

    def __init__(self, labels, dataset):
        """
        Labels indexes should match the indexes in the dataset.
        :param labels: Labels matching the data set
        :param dataset: Data of the images [[image1], [image2]...]
        """

        self.model = Sequential()

        self.predictions = []
        self.labels = labels
        self.dataset = dataset

        self.add_input_layer()
        self.add_hidden_layer()
        self.add_result_layer()

    def train(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'],
                           )

        self.model.fit(self.dataset, self.labels, epochs=self.EPOCHS, )

    def add_input_layer(self):
        """
        Same size as input number of pixels in input data
        :return:
        """
        # this converts our 3D feature maps to 1D feature vectors
        # Grab the first item of the dataset for the shape
        self.model.add(Flatten(input_shape=self.dataset[0].shape))

    def add_hidden_layer(self, num_of_nodes=128):
        """
        size is 15/20% of input size
        :return:
        """
        self.model.add(Dense(num_of_nodes, activation='relu'))

    def add_result_layer(self, num_of_nodes=10):
        # Softmax makes all results add up to 1
        self.model.add(Dense(num_of_nodes, activation='softmax'))

    def test(self, labels, data):
        test_loss, test_acc = self.model.evaluate(data, labels)
        print('Loss: {} \n Acc: {}'.format(test_loss, test_acc))

    def predict(self, data):
        self.predictions = self.model.predict(data)
        for pred in self.predictions:
            print('max prediciton: {}'.format(np.argmax(pred)))
