# Import section
from __future__ import print_function
# Supress warning and infomational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Keras is a library used for create Neural Network
import keras
# used for convert images and labels to numpy arrays
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D

# library used for split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split

# Matplotlib is a library used for data visualization
import matplotlib
import matplotlib.pyplot as plt
import csv

# NumPy is the fundamental package for scientific computing with Python
import numpy as np

# scikit-image is a collection of algorithms for image processing
import skimage.data
import skimage.transform

training_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
# test_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'

# function for load the images obtained from https://hackernoon.com/automatic-recognition-of-speed-limit-signs-deep-learning-with-keras-and-tensorflow-310d90af9826
# arguments: path to the traffic sign data,
# returns: list of images, list of corresponding labels


def load_dataset(root_path):
    labels = []
    images = []
    directories = [directory for directory in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, directory))]
    for directory in directories:
        label_directory = os.path.join(root_path, directory)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory) if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(directory))
    return images, labels

# self explanatory - from https://hackernoon.com/automatic-recognition-of-speed-limit-signs-deep-learning-with-keras-and-tensorflow-310d90af9826


def show_data_classes(images, labels):
    """Display the first image of each label."""
    miniatures = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in miniatures:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


load_images, load_labels = load_dataset(training_root_path)
# test_images, test_labels = load_dataset(test_root_path)

# Following method will print labels with miniature picture that we can verifi our label-image sets
# show_data_classes(load_images, load_labels)

# We need to normalize loaded images because neural network hads to got input of the same size.
# following loop is used to check what is common size of iumage in Dataset

# # Print some of images to check average sizes
# for single_image in load_images[70:90]:
#     print("shape: {0}, min: {1}, max: {2}".format(
#         single_image.shape, single_image.min(), single_image.max()))

# Output:
# shape: (34, 34, 3), min: 32, max: 255
# shape: (35, 35, 3), min: 22, max: 235
# shape: (37, 38, 3), min: 29, max: 235
# shape: (39, 39, 3), min: 19, max: 212
# shape: (39, 40, 3), min: 25, max: 208
# shape: (41, 41, 3), min: 19, max: 218
# shape: (44, 44, 3), min: 17, max: 224
# shape: (45, 46, 3), min: 18, max: 222
# shape: (47, 48, 3), min: 18, max: 248
# shape: (50, 51, 3), min: 17, max: 255
# shape: (52, 52, 3), min: 9, max: 255
# shape: (55, 55, 3), min: 10, max: 255
# shape: (59, 59, 3), min: 4, max: 255
# shape: (63, 62, 3), min: 8, max: 255
# shape: (70, 69, 3), min: 5, max: 255
# shape: (75, 74, 3), min: 6, max: 255
# shape: (85, 83, 3), min: 3, max: 255
# shape: (95, 92, 3), min: 0, max: 255
# shape: (108, 107, 3), min: 0, max: 255
# shape: (124, 122, 3), min: 0, max: 255


def normalize_images(load_images):
    normalized_images = []
    for image in load_images:
        normalized_images.append(skimage.transform.resize(image, (64, 64)))
    return normalized_images


normalized_training_images = normalize_images(load_images)
#normalized_test_images = normalize_images(test_images)

# # Print to double check
# for single_image in normalized_training_images[70:90]:
#     print("shape: {0}, min: {1}, max: {2}".format(
#         single_image.shape, single_image.min(), single_image.max()))

# Convert our images and labels to numpy arrays. So we can put it to NN
# convert class vectors to binary class matrices. One-hot encoding 
# 3 => 0 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
y_training = np.array(load_labels)
x_training = np.array(normalized_training_images)
# number of categories is related with directories count in training directory
number_of_categories = len([name for name in os.listdir(
    training_root_path) if os.path.isdir(os.path.join(training_root_path, name))])
y_training = to_categorical(y_training, number_of_categories)

inputs = Input(shape=(64,64,3))

#Split traininng data to measure accuracy of NN
x_train, x_test, y_train, y_test = train_test_split(x_training, y_training, test_size=0.25, random_state=42)

#Defining NN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), #convolution layer with 32 filters, if we increase it to 64 filters the accuracy will increase form 90 to 91%
                    activation='relu',
                    input_shape=inputs))
model.add(MaxPooling2D(pool_size=(2, 2))) # if we comment this line the accuracy will increase form 91 to 92% //removing early pooling layers
model.add(Conv2D(32, (3, 3), activation='relu')) # powtórzenie tej warstwy zgodnie ze schematem 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # flatten layer konwertuje output z naszej 2D sieci konwolucyjnej na 1D dla zwykłej sieci neuronowej
model.add(Dense(128, activation='relu')) # warstwa zwykłej sieci neuronowej
model.add(Dropout(0.5)) # warswta sieci dropout, zapobiega przeuczeniu sieci
model.add(Dense(number_of_categories, activation='softmax')) #ostatnia warstwa, odbiera wyniki i kategoryzuje do każdej ze zdefiniowanych klas

# define compile to minimize categorical loss, use ada delta optimized, and optimize to maximizing accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

# Train the model and test/validate the mode with the test data after each cycle (epoch) through the training data
# Return history of loss and accuracy for each epoch

# sizes of batch and # of epochs of data
batch_size = 128
epochs = 24

hist = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))

# Evaluate the model with the test data to get the scores on "real" data.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()