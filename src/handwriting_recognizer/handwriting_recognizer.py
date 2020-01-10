from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import imageio
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# save input image dimensions
img_rows, img_cols = 28, 28
image_index = 35
num_classes = 10
batch_size = 128
epochs = 10


# ========== Create training and test sets
def get_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    plt.imshow(x_train[image_index], cmap='Greys')
    plt.show()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


# =========== Create model
def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(img_rows, img_cols, 1)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("test_model.h5")


# ========= Test
# create_model(get_dataset())

im = imageio.imread("https://i.imgur.com/a3Rql9C.png")
gray = np.dot(im[..., :3], [0.299, 0.587, 0.114])
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.show()
# reshape the image
gray = gray.reshape(1, img_rows, img_cols, 1)

# normalize image
gray = gray / 255

# load the model

model = load_model("test_model.h5")

# predict digit
prediction = model.predict(gray)
print(prediction.argmax())
