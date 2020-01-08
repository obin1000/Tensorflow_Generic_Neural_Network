from generic_neural_network import Dataset, Learner
import numpy as np

# Example with a train set from Google and a test set from the filesystem
IMAGE_DIMENSIONS = 400
# ====== Create Train Set =======
train_set = Dataset()
# 0 is cat, 1 is dog
train_set.add_category([0, 4000, 'happy face'])
train_set.add_category([1, 4000, 'sad face'])
train_set.IMAGE_DIMENSIONS = IMAGE_DIMENSIONS
train_set.run()
train_set.export_compressed(train_set.relative_to_absolute('../../dataset/'))
train_labels, train_data = train_set.get_shuffled_label_data_separated()

# ====== Create Test Set =======
test_set = Dataset()
# 0 is cat, 1 is dog
test_set.add_category([0, 0, train_set.relative_to_absolute('../../dataset/test/happy/')])
test_set.add_category([1, 0, train_set.relative_to_absolute('../../dataset/test/sad/')])
test_set.IMAGE_DIMENSIONS = IMAGE_DIMENSIONS
test_set.run()
test_labels, test_data = test_set.get_shuffled_label_data_separated()

# ======= Train a network ======
genie = Learner(np.asarray(train_labels), np.asarray(train_data), len(train_set.get_categories()))
genie.EPOCHS = 20
genie.train()

# ======= Test the network =======

genie.predict(np.asarray(test_data), labels=test_labels)
