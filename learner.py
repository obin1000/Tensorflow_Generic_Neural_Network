import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import cv2
import os


class Learner:
    EPOCHS = 10  # Number of times to see the dataset

    def __init__(self, labels, datase):
        """
        Labels indexes should match the indexes in the dataset.
        :param labels:
        :param dataset: Data of the images [[image1], [image2]...]
        """
        for d in range(len(labels)):
            if labels[d] == 'dog':
                labels[d] = 0
            else:
                labels[d] = 1
        plt.imshow(datase[30], cmap=plt.cm.binary)
        plt.show()
        self.model = Sequential()
        self.labels = labels
        self.dataset = datase

        self.add_input_layer()
        self.add_hidden_layer()
        self.add_value_layer()

    def train(self):
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit(self.dataset, self.labels, epochs=self.EPOCHS)

    def add_input_layer(self):
        """
        Same size as input number of pixels in input data
        :return:
        """
        self.model.add(Flatten(input_shape=self.dataset[0].shape))  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))  # Make all results add up to 1

    def add_hidden_layer(self):
        """
        15/20% of input size
        :return:
        """
        pass

    def add_value_layer(self):
        pass

    def add_layers(self):
        pass

    def add_training_data(self):
        pass

    def test(self, labels, data):
        test_loss, test_acc = self.model.evaluate(labels, data)
        print('Loss: {} \n Acc: {}'.format(test_loss, test_acc))
