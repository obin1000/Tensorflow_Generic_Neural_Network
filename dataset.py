import os
import cv2
import random
import numpy as np


# TODO: Balance the dataset
class Dataset:
    """
    Helps creating you a dataset to use for a neural network. It can do multiple things:
    Provide an path on the local system with images. This class will read them with Opencv, turn them to greyscale and
        resize them to the same dimensions. After this, they are ready to be used by Tensorflow.
    """
    IMAGE_SIZE = 100
    MAX_BALANCE_DIFFERENCE = 5

    fileCounter = 0

    def __init__(self):
        """
        Init the class variables
        Categories: Expects a list. The first index should be the name of the category. The rest of the list you can
            add sources for the dataset for this category. This way you can add as many data sources as you want.
            Only accepts full paths.
            e.g: ['Dog', '/tmp/dogs/', 'X:/images/dogs/']
        Data: The data from the sources, mapped by category in the following format:
            [['Cat', data 1, data 2...],
             ['Dog', data 1, data 2...],
             ...
            ]
        """
        self.Categories = []
        self.Data = []

    def add_category(self, cat):
        """
        Add a category to the dataset
        :param cat: The category to add, should look like: ['Name', 'Source1', 'Source2'...]
        :return: None
        """
        self.Categories.append(cat)

    def run(self):
        # Grab each category and parse the data from given sources
        for category in self.Categories:

            name = category[0]
            # Parse the data sources, skip the first item since that is the name of the category
            for source in category[1:]:
                if source.startswith('/'):
                    self._run_filesystem(source, name)
                elif source.startswith('http://') or source.startswith('https://'):
                    if source.contains('www.google'):
                        self._run_google_images()
                    else:
                        self._run_online()

                print('Parsed {} items for category {} from {}'.format(len(self.Data[-1]), name, source))

    def normalize_data(self, data, max_value=255):
        """
        Normalize some data by changing its values to values between 0 and 1
        :param data: Data to be normalized (img)
        :param max_value: Highest value occurring in the dataset
        :return: Data, but normalized
        """
        return data / max_value

    def balance_data(self, remove=True):
        """
        Check for balance in the dataset. Preferably all categories have exactly the same number of data samples.
        If there is an imbalance, data will be removed from the biggest set until the balance is within margins.
        :return: None
        """
        pass

    def relative_to_absolute(self, url):
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, url)

    def _run_filesystem(self, dir, category, normalize=True):
        category_data = [category]

        for image in os.listdir(dir):
            try:
                # Use opencv to read the image. In grayscale for less memory usage and easier parsing
                img_values = cv2.imread(os.path.join(dir, image), cv2.IMREAD_GRAYSCALE)
                # Normalize shape of image to given dimensions and values of grayscale to values between 0 and 1.
                if normalize:
                    img_values = cv2.resize(img_values, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                    img_values = self.normalize_data(img_values)
                category_data.append(img_values)
            except Exception as e:
                # Skip all broken images
                print('Something went wrong with image {}: {}'.format(image, e))
        self.Data.append(category_data)

    def _run_online(self):
        pass

    def _run_google_images(self):
        pass

    def get_label_data_separated(self):
        """
        Get the labels and data separated from each other like ['Cat','Cat','Dog'], [data, data, data...]
        :return: The labels and data separated in two different arrays
        """
        # Arrays for labels and matching data
        labels = []
        datas = []
        for category in self.Data:
            name = category[0]
            for data in category[1:]:
                labels.append(name)
                datas.append(data)
        return labels, datas

    def get_labeled_data(self):
        """
        Get the data labeled in order: [['Cat', data], ['Cat', data] ... ['Dog', data], ['Dat', data] ...]
        :return: The data labeled by category
        """
        labeled_data = []
        # Copy the data to the random data with labels as [category name, data][category name, data]...
        for category in self.Data:
            name = category[0]
            for data in category[1:]:
                labeled_data.append([name, data])

        return labeled_data

    def get_random_data(self):
        """
        Has labeled shuffled data: [['Cat', data], ['Dog', data], ['Dog', data], ['Cat', data] ...]
        :return: The data shuffled and labeled by category
        """
        random_data = self.get_labeled_data()
        # Shuffle dataset to balance the number of occurrences of the categories.
        # Not 100 cats followed by 100 dogs, that's bad
        random.shuffle(random_data)
        return random_data

    def export_bin(self, path):
        """
        Export the data to a csv file
        :param path: Path to the save location
        :return: None
        """
        np.save('{}data{}.npy'.format(path, Dataset.fileCounter), self.Data)
        Dataset.fileCounter += 1

    def export_compressed(self, path):
        """
        Save the data compressed as .npz file
        :param path: Path to the save location
        :return: None
        """
        np.savez_compressed('{}data{}.npz'.format(path, Dataset.fileCounter), self.Data)
        Dataset.fileCounter += 1

    def import_bin(self, path):
        """
        Will override current data
        :param path: Path to the csv file
        :return: None
        """
        # Allow pickle to be able to import an array
        self.Data = np.load(path, allow_pickle=True)

    def import_compressed(self, path, array_num='arr_0'):
        self.Data = np.load(path, allow_pickle=True)[array_num]

    def __str__(self):
        cats = ''
        for cat in self.Data:
            cats += '{}: {}\n'.format(cat[0], len(cat))
        return 'Categories: {} \n Total data: {} \n {}'.format(self.Categories, len(self.Data), cats)
