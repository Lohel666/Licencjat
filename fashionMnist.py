# Train a ConvNet on the MNIST fashion data. This data consists of 10 classes of fashion images such
# as shorts, dresses, shoes, purses, etc. These images replace the handwritten digits in the classic MNIST dataset.
# This chnages makes it harder to get a high score and more coslely reflects real world usage of images
# classification. And at the same time, is still small enough for the average PC to train in a short time.
# See https://github.com/zalandoresearch/fasion-mnist for information and code on Fashion MNIST

# This code is based on MNIST example found at Keras.io

from __future__ import print_function
import keras
from keras.datasets import fashion_mnist  # new with Keras 2.1.2. Yah!
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Supress warning and infomational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Number of classes - do not change unless the data changes
num_classes = 10  # shorts, dresses, shoes, purses, etc

# sizes of batch and # of epchos of data
batch_size = 128
epchos = 24

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Deal with format issues between different backends. Some put the # of channels in the image before the width and height
if K.image_data_format() == 'cannels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = X_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = X_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Type convert and scale the test and training data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices. One-hot encoding 
# 3 => 0 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), #convolution layer with 32 filters
                    activation='relu',
                    input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu')) # powtórzenie tej warstwy zgodnie ze schematem 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # flatten layer konwertuje output z naszej 2D sieci konwolucyjnej na 1D dla zwykłej sieci neuronowej
model.add(Dense(128, activtion='relu')) # warstwa zwykłej sieci neuronowej
model.add(Dropout(0.5)) # warswta sieci dropout, zapobiega przeuczeniu sieci
model.add(Dropout(num_classes, activation='softmax')) #ostatnia warstwa, odbiera wyniki i kategoryzuje do każdej ze zdefiniowanych 10 klas

# define compile to minimize categorical loss, use ada delta optimized, and optimize to maximizing accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
                