import os
import cv2
import random
from numpy import asarray, save, savez_compressed, load


# TODO: Balance the dataset
class Dataset:
    """
    Helps creating you a dataset to use for a neural network. It can do multiple things:
    Provide an path on the local system with images. This class will read them with Opencv, turn them to greyscale and
        resize them to the same dimensions. After this, they are ready to be used by Tensorflow.
    """
    IMAGE_SIZE = 4

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
        RandomData: Has labeled shuffled data:
            [['Cat', data], ['Dog', data], ['Dog', data], ['Cat', data] ...]

        """
        self.Categories = []
        self.Data = []
        self.RandomData = []

    def add_category(self, cat):
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

    def create_random_data(self):
        # Copy the data to the random data as [category name, data][category name, data]...
        for category in self.Data:
            name = category[0]
            for data in category[1:]:
                self.RandomData.append([name, data])

        # Shuffle dataset to balance the number of occurrences of the categories.
        # Not 100 cats followed by 100 dogs, that's bad
        random.shuffle(self.RandomData)

    def relative_to_absolute(self, url):
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, url)

    def _run_filesystem(self, dir, category):
        category_data = [category]

        for image in os.listdir(dir):
            try:
                # Use opencv to read the image in grayscale for less memory usage and easier parsing
                img_values = cv2.imread(os.path.join(dir, image), cv2.IMREAD_GRAYSCALE)
                # Change the shape of the image
                values_resized = cv2.resize(img_values, (Dataset.IMAGE_SIZE, Dataset.IMAGE_SIZE))
                category_data.append(values_resized)
            except Exception as e:
                # Skip all broken images
                print('Something went wrong: ' + str(e))
        self.Data.append(category_data)

    def _run_online(self):
        pass

    def _run_google_images(self):
        pass

    def export_bin(self, path):
        """
        Export the data to a csv file
        :param path: Path to the save location
        :return: None
        """
        save('{}data{}.npy'.format(path, Dataset.fileCounter), asarray(self.Data))
        Dataset.fileCounter += 1

    def export_compressed(self, path):
        """
        Save the data compressed as .npz file
        :param path: Path to the save location
        :return: None
        """
        savez_compressed('{}data{}.npz'.format(path, Dataset.fileCounter), asarray(self.Data))
        Dataset.fileCounter += 1

    def import_bin(self, path):
        """
        Will override current data
        :param path: Path to the csv file
        :return: None
        """
        # Allow pickle to be able to import an array
        self.Data = load(path, allow_pickle=True)
        self.create_random_data()

    def import_compressed(self, path, array_num='arr_0'):
        self.Data = load(path, allow_pickle=True)[array_num]
        self.create_random_data()

    def __str__(self):
        return 'Data: {}'.format(self.Data)
