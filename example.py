from dataset import Dataset

# Cats and dogs dataset from Microsoft : https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

# Create the dataset
set = Dataset()

# Add categories to the dataset with their sources. In this case we use the categories cats and dogs
# and refer to images of cats and dogs on the filesystem
cat_path = set.relative_to_absolute('dataset/cats_and_dogs/Cat/')
dog_path = set.relative_to_absolute('dataset/cats_and_dogs/Dog/')
cat = ['cat', cat_path]
dog = ['dog', dog_path]
set.add_category(cat)
set.add_category(dog)

# Set the resolution of the images
set.IMAGE_SIZE = 100

# Parse the data from the sources into the dataset
set.run()

# Export the dataset to a file, compressed or binary
# set.export_bin(set.relative_to_absolute('dataset/'))
set.export_compressed(set.relative_to_absolute('dataset/'))

# Import a dataset from a file
# set.import_bin(set.relative_to_absolute('dataset/data0.npy'))
# set.import_compressed(set.relative_to_absolute('dataset/data0.npz'))
