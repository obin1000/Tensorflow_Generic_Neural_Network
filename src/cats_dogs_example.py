from generic_neural_network import Dataset, Learner
import numpy as np

# Example with a train set from the local filesystem and a test set from Google
IMAGE_SIZE = 200
# ====== Create Train Set =======
train_set = Dataset()
# 0 is cat, 1 is dog
train_set.add_category([0, 0, train_set.relative_to_absolute('../../dataset/Cat/')])
train_set.add_category([1, 0, train_set.relative_to_absolute('../../dataset/Dog/')])
Dataset.IMAGE_SIZE = IMAGE_SIZE
train_set.run()
train_labels, train_data = train_set.get_shuffled_label_data_separated()


# ====== Create Test Set =======
test_set = Dataset()
# 0 is cat, 1 is dog
test_set.add_category([0, 0, train_set.relative_to_absolute('../../dataset/sCat/')])
test_set.add_category([1, 0, train_set.relative_to_absolute('../../dataset/sDog/')])
# test_set.add_category([0, 10, 'cat', 'kat', 'poes'])
# test_set.add_category([1, 10, 'dog', 'hond', 'puppy'])
test_set.run()
test_labels, test_data = test_set.get_shuffled_label_data_separated()

# ======= Train a network ======
print('shape: {} {} {}'.format(len(train_data), len(train_data[0]), len(train_data[0][0])))
genie = Learner(train_labels, train_data, len(train_set.get_categories()))
genie.EPOCHS = 10
genie.train()

# ======= Test the network =======

genie.predict(test_data, labels=test_labels)


# max prediciton: 0 correct answer: 0
# max prediciton: 1 correct answer: 1
# max prediciton: 0 correct answer: 0
# max prediciton: 1 correct answer: 0
# max prediciton: 0 correct answer: 0
# max prediciton: 0 correct answer: 0
# max prediciton: 1 correct answer: 1
# max prediciton: 1 correct answer: 1
# max prediciton: 0 correct answer: 1
# max prediciton: 1 correct answer: 1
# max prediciton: 0 correct answer: 0
# max prediciton: 0 correct answer: 0
# max prediciton: 1 correct answer: 1
# max prediciton: 1 correct answer: 1
# max prediciton: 1 correct answer: 1
# max prediciton: 0 correct answer: 1
# max prediciton: 0 correct answer: 0
# max prediciton: 1 correct answer: 0
# max prediciton: 0 correct answer: 0
# max prediciton: 1 correct answer: 1
# max prediciton: 1 correct answer: 1
# max prediciton: 0 correct answer: 0
