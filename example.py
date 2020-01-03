from dataset import Dataset
import numpy as np
from learner import Learner

# Cats and dogs dataset from Microsoft : https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

# ============ Create dataset =================
# Create the dataset
train_set = Dataset()

# Add categories to the dataset with their sources. In this case we use the categories cats and dogs
# and refer to images of cats and dogs on the filesystem
cat_path = train_set.relative_to_absolute('dataset/cats_and_dogs/sCat/')
dog_path = train_set.relative_to_absolute('dataset/cats_and_dogs/sDog/')
cat = ['cat', cat_path]
dog = ['dog', dog_path]
train_set.add_category(cat)
train_set.add_category(dog)

# Set the resolution of the images
train_set.IMAGE_SIZE = 40

# Parse the data from the sources into the dataset
train_set.run()

# Export the dataset to a file, compressed or binary
# train_set.export_bin(set.relative_to_absolute('dataset/'))
# train_set.export_compressed(set.relative_to_absolute('dataset/s'))

# Import a dataset from a file
# train_set.import_bin(set.relative_to_absolute('dataset/data0.npy'))
# train_set.import_compressed(set.relative_to_absolute('dataset/sdata0.npz'))

labels, data = train_set.get_label_data_separated()
# npdata = np.reshape(np.asarray(data), (set.IMAGE_SIZE, set.IMAGE_SIZE))


# ============ Learn ==============

genie = Learner(labels, np.asarray(data))

genie.EPOCHS = 10
genie.train()

# ============ Test =============
# Create the dataset
test_set = Dataset()

# Add categories to the dataset with their sources. In this case we use the categories cats and dogs
# and refer to images of cats and dogs on the filesystem
test_cat = train_set.relative_to_absolute('dataset/cats_and_dogs/tCat/')
test_dog = train_set.relative_to_absolute('dataset/cats_and_dogs/tDog/')
tcat = ['cat', test_cat]
tdog = ['dog', test_dog]
test_set.add_category(tcat)
test_set.add_category(tdog)

test_set.run()
da, ta = test_set.get_label_data_separated()
print(da)
print(ta)
genie.test(da, ta)
