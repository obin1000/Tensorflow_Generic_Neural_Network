
from dataset import Dataset

set = Dataset()

# Cats and dogs dataset from Microsoft : https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

# cat_path = set.relative_to_absolute('dataset/cats_and_dogs/sCat/')
# dog_path = set.relative_to_absolute('dataset/cats_and_dogs/sDog/')
#
# cat = ['cat', cat_path]
# dog = ['dog', dog_path]
#
# set.add_category(cat)
# set.add_category(dog)
#
# set.run()
#
# set.export_bin(set.relative_to_absolute('dataset/'))
# set.export_compressed(set.relative_to_absolute('dataset/'))

# Import dataset from binary file
# print(set)
# set.import_bin(set.relative_to_absolute('dataset/data0.npy'))
# print(set)
# Import dataset from compressed file
print(set)
set.import_compressed(set.relative_to_absolute('dataset/data1.npz'))
print(set)
