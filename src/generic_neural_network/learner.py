import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np


class Learner:
    EPOCHS = 10  # Number of times to see the dataset
    LEARNING_RATE = 0.05
    NUM_NODES_LAYER1 = 128
    NUM_NODES_LAYER2 = 10

    def __init__(self, labels, dataset, num_of_categories):
        """
        Labels indexes should match the indexes in the dataset.
        :param labels: Labels matching the data set
        :param dataset: Data of the images [[image1], [image2]...]
        """

        self.model = Sequential()
        self.predictions = []
        self.labels = labels
        print(len(dataset))
        self.dataset = dataset
        self.num_of_categories = num_of_categories

        # Binary data is broken atm, so always use multi
        if num_of_categories <= -2:
            print('Using binary model')
            self.get_binary_model()
        else:
            print('Using multi model')
            self.get_multi_model()

    def train(self):
        # Binary data is broken atm, so always use multi
        if self.num_of_categories <= -2:
            print('Using binary compiler')
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            print('Using multi compiler')
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print('dataset shape: {}'.format(self.dataset.shape))
        print(len(self.dataset))
        print(len(self.labels))
        self.model.fit(self.dataset, self.labels, epochs=self.EPOCHS)

    def get_binary_model(self):
        self.model.add(Conv2D(16, kernel_size=2, activation='relu', input_shape=self.dataset[0].shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(1, activation='sigmoid'))

    def get_multi_model(self):
        # Same size as input number of pixels in input data
        # this converts our 3D feature maps to 1D feature vectors
        # Grab the first item of the dataset for the shape
        self.model.add(Flatten(input_shape=self.dataset[0].shape))
        # size is 15/20% of input size
        self.model.add(Dense(self.NUM_NODES_LAYER1, activation='relu'))
        # Softmax makes all results add up to 1
        self.model.add(Dense(self.num_of_categories, activation='softmax'))

    def test(self, labels, data):
        test_loss, test_acc = self.model.evaluate(data, labels)
        print('Loss: {} \n Acc: {}'.format(test_loss, test_acc))

    def predict(self, data, labels=None):
        print('Test shape: {}'.format(data.shape))
        self.predictions = self.model.predict(data)
        counter = 0
        for pred in self.predictions:
            if labels is None:
                print('max prediciton: {}'.format(np.argmax(pred)))
            else:
                print('max prediciton: {} correct answer: {}'.format(np.argmax(pred), labels[counter]))
            counter += 1
