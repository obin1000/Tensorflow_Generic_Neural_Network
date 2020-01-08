from generic_neural_network import Dataset, Learner
import numpy as np

# Example with a train set from Google and a test set from Google
IMAGE_SIZE = 200
# ====== Create Train Set =======
train_set = Dataset()
# 0 is cat, 1 is dog
train_set.add_category([0, 20, 'happy face'])
train_set.add_category([1, 20, 'sad face'])
Dataset.IMAGE_SIZE = IMAGE_SIZE
train_set.run()
train_labels, train_data = train_set.get_shuffled_label_data_separated()


# ====== Create Test Set =======
test_set = Dataset()
# 0 is cat, 1 is dog
test_set.add_category([2, 10, 'blije smiley'])
test_set.add_category([3, 10, 'verdrietige smiley'])
test_set.run()
test_labels, test_data = test_set.get_shuffled_label_data_separated()

# ======= Train a network ======
genie = Learner(np.asarray(train_labels), np.asarray(train_data), len(train_set.get_categories()))
genie.EPOCHS = 10
genie.train()

# ======= Test the network =======

genie.predict(np.asarray(test_data), labels=test_labels)
