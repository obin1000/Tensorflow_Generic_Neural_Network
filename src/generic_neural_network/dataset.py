import os
import cv2
import shutil
import random
import datetime
import numpy as np
from google_images_download import google_images_download


# TODO: Balance the dataset
class Dataset:
    """
    Helps creating you a dataset to use for a neural network. It can do multiple things:
    Provide an path on the local system with images. This class will read them with Opencv, turn them to greyscale and
        resize them to the same dimensions. After this, they are ready to be used by Tensorflow.
    This class can also generate a dataset it self. All you need to do is provide it with topics related to the category
    It will use the topics to collect data from google images. You'll need to provide multiple topics, since the library
    is only able to collect 100 images per topic at max.
    """
    IMAGE_DIMENSIONS = 100
    # Max difference in size of the dataset in percent (scale 0 to 1)
    MAX_DATA_BALANCE_DIFFERENCE = 0.10
    DEFAULT_DATA_DIRECTORY = '../../dataset/download/'
    MAX_GOOGLE_IMAGES_PER_SEARCH = 100
    IMAGES_EXTENSION = ['.png', '.jpg', '.jpeg', '.webp']
    # 'past-year', 'past-month', 'past-7-days', 'past-24-hours'
    GOOGLE_FILTERS_TIME = ['past-year', 'past-month', 'past-7-days', 'past-24-hours']
    # Possibilities are: , 'clip-art', 'face', 'line-drawing', 'animated'
    GOOGLE_FILTERS_TYPE = ['photo']
    # 'square', 'tall', 'wide', 'panoramic'
    GOOGLE_FILTERS_ASPECT_RATIO = ['square', 'tall', 'wide', 'panoramic']

    def __init__(self):
        """
        Init the class variables
        Categories: Expects a list. The first index should be the name of the category. The rest of the list you can
            add sources for the dataset for this category. This way you can add as many data sources as you want.
            Only accepts full paths.
            e.g: ['Dog', '/tmp/dogs/', 'X:/images/dogs/']
        Data: The data from the sources, mapped by category in a dictionary:
            {'Cat' : [data 1, data 2...],
             'Dog' : [data 1, data 2...],
             ...
            }
        """
        self.categories = []
        self.Data = {}

    def add_category(self, cat):
        """
        Add a category to the dataset
        :param cat: The category to add, should look like: ['Name',size ,'Source1', 'Source2'...]
        :return: None
        """
        # Add the category to the dictionary with an empty list to store the data
        self.Data[cat[0]] = []
        self.categories.append(cat)

    def get_categories(self):
        """
        Get the current categories
        :return: The categories
        """
        return self.categories

    def run(self, balance=True):
        """
        Parse given categories to a dataset.
        :param balance: Balance the dataset after parsing?
        :return: None
        """
        # Grab each category and parse the data from given sources
        for category in self.categories:
            # Sources meant for google search
            google_sources = []
            # Name of the category
            name = category[0]
            size = category[1]

            # Parse the data sources, skip the first item since that is the name of the category
            for source in category[2:]:
                if ('/' in source) or ('\\' in source):
                    self.Data[name] += self._run_filesystem(source)

                elif not ('www.' in source):
                    google_sources.append(source)

                else:
                    print('Could not understand source: {}'.format(source))

            if len(google_sources):
                # Google cannot handle individual sources, but needs them as list.
                print('Generating dataset from Google of {}: {}'.format(category, google_sources))
                self.Data[name] += self._run_google_images(google_sources, size)

        if balance:
            self.balance_data()

    def _run_filesystem(self, dir):
        """
        Recursive scans given directory for images for given category with default parsing parameters.
        :param dir: Directory with data.
        :return: A list with all images parsed for given directory
        """
        gathered_data = []
        for file in os.listdir(dir):
            full_path = os.path.join(dir, file)
            filename, file_extension = os.path.splitext(file)
            if os.path.isdir(full_path):
                gathered_data += self._run_filesystem(full_path)
            elif file_extension.lower() in self.IMAGES_EXTENSION:
                gathered_data.append(self.read_image(full_path))
            else:
                print('Unknown file type found: {}'.format(file_extension))
        return gathered_data

    def _run_google_images(self, topics, num_of_images, remove_after_download=True):
        """
        Creates a dataset from Google images.
        :param topics: A list of topics to search  the web with
        :param num_of_images: Number of images wanted in the dataset
        :return: A list with images parsed for given topics
        """

        # class instantiation
        response = google_images_download.googleimagesdownload()

        # Group images by category to prevent double images in the dataset
        output_dir = self.relative_to_absolute(os.path.join(self.DEFAULT_DATA_DIRECTORY, self.get_filename()))

        # Fill the output directory with images of given topics.
        for topic in topics:
            for filter_type in self.GOOGLE_FILTERS_TYPE:
                for aspect_ratio in self.GOOGLE_FILTERS_ASPECT_RATIO:
                    for time in self.GOOGLE_FILTERS_TIME:
                        # Set the number of images limit per search
                        num_img_per_search = self.MAX_GOOGLE_IMAGES_PER_SEARCH
                        if num_of_images < self.MAX_GOOGLE_IMAGES_PER_SEARCH:
                            num_img_per_search = num_of_images

                        # Decided not to use a prefix in the file name. This way images that appear at
                        # multiple searching will not be used multiple times in the dataset,
                        # but will just overwrite itself.
                        # Creating list of arguments for a Google search
                        arguments = {'keywords': topic, 'limit': num_img_per_search, 'no_directory': True,
                                     'output_directory': output_dir, 'type': filter_type,
                                     'time': time, 'aspect_ratio': aspect_ratio}

                        # Start the download
                        paths = response.download(arguments)

                        # check is enough data is gathered
                        current_images = len(os.listdir(output_dir))
                        print('Download at {}/{}'.format(current_images, num_of_images))
                        if current_images > num_of_images:
                            break
                    # check is enough data is gathered
                    if len(os.listdir(output_dir)) > num_of_images:
                        break
                # This is just bullshit
                if len(os.listdir(output_dir)) > num_of_images:
                    break
            # check is enough data is gathered
            if len(os.listdir(output_dir)) > num_of_images:
                break

        # Use filesystem to parse and add the images to the dataset
        data = self._run_filesystem(output_dir)

        #  Cleanup: Remove downloaded files after parsing it
        if remove_after_download:
            shutil.rmtree(output_dir, ignore_errors=True)

        return data

    def normalize_image(self, data, max_value=255):
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
        print('Starting balancing data')
        lengths = {}
        smallest_length = 9999999999999999.9
        for cat in self.Data:
            # Grab the length of each category
            lengths[cat] = len(self.Data.get(cat))

        for cat in lengths:
            # Get the smallest length
            length = lengths.get(cat)
            if length < smallest_length:
                smallest_length = length

        print('Smallest set is {}'.format(smallest_length))

        for cat in lengths:
            # Calculate the percent difference of each category with the smallest category
            length = lengths.get(cat)
            difference = (1.0 - (smallest_length / length))
            if difference > self.MAX_DATA_BALANCE_DIFFERENCE:
                print('Unacceptable difference in set {}'.format(cat))
                if remove:
                    # Trim the category to the size of the smallest category
                    trim_length = smallest_length + (smallest_length * self.MAX_DATA_BALANCE_DIFFERENCE)
                    del self.Data.get(cat)[trim_length:]

        new_lengths = {}
        for cat in self.Data:
            # Grab the length of each category
            new_lengths[cat] = len(self.Data.get(cat))

        print('Before: {} \n After: {}'.format(lengths, new_lengths))

    def relative_to_absolute(self, path):
        """
        Convert a path relative to this file to an absolute path.
        :param path: Path relative to this file.
        :return: Absolute path
        """
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, path)

    def read_image(self, path, color=cv2.IMREAD_GRAYSCALE, normalize=True, resize=True):
        """
        Use opencv to read an image from the file system and parse it.
        :param path: Path to the image.
        :param color: Color to convert the image to.
        :param normalize: Normalize the data? (scale values to values between 0 and 1)
        :param resize: Resize images to the default dimensions?
        :return: A nested list with pixel values of the image
        """
        img_values = []
        try:
            # Use opencv to read the image
            img_values = cv2.imread(path, color)
            # Normalize shape of image to given dimensions and values of grayscale to values between 0 and 1.
            if resize:
                img_values = cv2.resize(img_values, (self.IMAGE_DIMENSIONS, self.IMAGE_DIMENSIONS))
            if normalize:
                img_values = self.normalize_image(img_values)
        except Exception as e:
            # Skip all broken images
            print('Something went wrong with image {}: {}'.format(path, e))

        return img_values

    def get_label_data_separated(self):
        """
        Get the labels and data separated from each other like ['Cat','Cat','Dog'], [data, data, data...]
        :return: The labels and data separated in two different arrays
        """
        # Arrays for labels and matching data
        labels = []
        datas = []
        for category in self.Data:
            for data in self.Data[category]:
                labels.append(category)
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
            for data in self.Data[category]:
                labeled_data.append([category, data])

        return labeled_data

    def get_shuffled_label_data_separated(self):
        """
        First shuffles data, then separates the labels from the data. ['Cat','Dog','Cat'], [data, data, data...]
        :return: The labels and data
        """
        random_data = self.get_shuffled_data()
        labels = []
        datas = []
        # Convert [['Dog', data]] to ['Dog'] [data]
        for data in random_data:
            labels.append(data[0])
            datas.append(data[1])
        return labels, datas

    def get_shuffled_data(self):
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
        print('exporting dataset to {}'.format(path))
        np.save('{}{}.npy'.format(path, self.get_filename()), self.Data)

    def export_compressed(self, path):
        """
        Save the data compressed as .npz file
        :param path: Path to the save location
        :return: None
        """
        print('exporting dataset to {}'.format(path))
        np.savez_compressed('{}{}.npz'.format(path, self.get_filename()), self.Data)

    def import_bin(self, path):
        """
        Will override current data
        :param path: Path to the file
        :return: None
        """
        # Allow pickle to be able to import an array
        self.Data = np.load(path, allow_pickle=True).item()
        self.categories = self.Data.keys()

    def import_compressed(self, path, array_num='arr_0'):
        """
        Read a numpy compressed file
        :param path: Path to the file
        :param array_num: Num ber of the array to grab from the compressed file
        :return: None
        """
        self.Data = np.load(path, allow_pickle=True)[array_num].item()
        self.categories = self.Data.keys()

    def get_filename(self):
        """
        Get a unique filename
        :return: A unique filename
        """
        return str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')

    def __str__(self):
        cats = ''
        total = 0
        for cat in self.Data:
            num = len(self.Data[cat])
            cats += '{}: {}\n'.format(cat, num)
            total += num
        return 'Categories: {} \n Total data: {} \n {}'.format(self.categories, total, cats)
