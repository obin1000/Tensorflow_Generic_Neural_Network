from dataset import Dataset
import numpy as np
from learner import Learner

# Cats and dogs dataset from Microsoft : https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

# ============ Create dataset =================
# Create the training dataset
train_set = Dataset()

# Add categories to the dataset with their sources. In this case we use the categories cats and dogs
# and refer to images of cats and dogs on the filesystem
cat_path = train_set.relative_to_absolute('dataset/cats_and_dogs/Cat/')
dog_path = train_set.relative_to_absolute('dataset/cats_and_dogs/Dog/')
cat = [0, cat_path]
dog = [1, dog_path]
train_set.add_category(cat)
train_set.add_category(dog)

# Add category to collect a dataset from google images
# train_set.add_category(['dog', 'dog', 'hond', 'puppy'])
# train_set.add_category(['cat', 'cat', 'kat', 'poes'])

# Set the resolution of the images
train_set.IMAGE_SIZE = 28

# Parse the data from the sources into the dataset
train_set.run()

# Export the dataset to a file, compressed or binary
# train_set.export_bin(train_set.relative_to_absolute('dataset/s'))
train_set.export_compressed(train_set.relative_to_absolute('dataset/s'))

# Import a dataset from a file
# train_set.import_bin(train_set.relative_to_absolute('dataset/sdata0.npy'))
# train_set.import_compressed(train_set.relative_to_absolute('dataset/sdata0.npz'))

train_labels, train_data = train_set.get_shuffled_label_data_separated()

# ============ Learn ==============
# Tensorflow expects the data to be a numpy array
genie = Learner(train_labels, np.asarray(train_data))

genie.EPOCHS = 10
genie.train()

# ============ Test =============
# Create the test dataset
test_set = Dataset()

test_set.IMAGE_SIZE = 28

# Search Google images for test images
tcat = [0, 'cat', 'kitten']
tdog = [1, 'dog', 'puppy']
test_set.add_category(tcat)
test_set.add_category(tdog)

test_set.run()
test_labels, test_data = test_set.get_shuffled_label_data_separated()

genie.predict(np.asarray(test_data))
print('Expected result: {}'.format(test_labels))
