""" Import section """
from __future__ import print_function
import normalize_data
"""Supress warning and infomational messages"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""Keras is a library used for create Neural Network"""
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
"""used for convert images and labels to numpy arrays"""
from keras.utils.np_utils import to_categorical

"""library used for split arrays or matrices into random train and test subsets"""
from sklearn.model_selection import train_test_split

"""Matplotlib is a library used for data visualization"""
import matplotlib
import matplotlib.pyplot as plt
import csv

"""NumPy is the fundamental package for scientific computing with Python"""
import numpy as np

training_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
# test_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'

load_images, load_labels = normalize_data.load_dataset(training_root_path)
# test_images, test_labels = load_dataset(test_root_path)

normalized_training_images = normalize_data.normalize_images(load_images)
# normalized_test_images = normalize_images(test_images)

"""
# Print to double check
for single_image in normalized_training_images[70:90]:
    print("shape: {0}, min: {1}, max: {2}".format(
        single_image.shape, single_image.min(), single_image.max()))
"""

"""
Convert our images and labels to numpy arrays. So we can put it to NN
convert class vectors to binary class matrices. One-hot encoding 
3 => 0 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0
"""
y_training = np.array(load_labels)
x_training = np.array(normalized_training_images)
""" number of categories is related with directories count in training directory """
number_of_categories = len([name for name in os.listdir(
    training_root_path) if os.path.isdir(os.path.join(training_root_path, name))])
y_training = to_categorical(y_training, number_of_categories)

"""Split traininng data to measure accuracy of NN"""
x_train, x_validation, y_train, y_validation = train_test_split(x_training, y_training, test_size=0.25, random_state=42)

"""
Defining NN
"""
model = Sequential()
"""convolution layer with 32 filters"""
model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu'))
"""early pooling layers"""
# model.add(MaxPooling2D(pool_size=(2, 2))) # removal of first pooling layer causes increase of accuracy
"""2nd convolutional layer with pooling layer"""
model.add(Conv2D(32, (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2)))
"""flatten layer converts output from 2D Convolutional Neural Network to 1D for 'normal' Neural Network """
model.add(Flatten())
"""NN"""
model.add(Dense(128, activation='relu'))
"""dropout layer, prevents network overfitting, percentage chance to change output to 0"""
model.add(Dropout(0.5))
"""final layer, collects results and fits them to every defined class"""
model.add(Dense(number_of_categories, activation='softmax')) 

"""define compile to minimize categorical loss, use ada delta optimized, and optimize to maximizing accuracy"""
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

"""
Train the model and test/validate the mode with the test data after each cycle (epoch) through the training data
Return history of loss and accuracy for each epoch
"""

"""
Parameters to fit the NN. 
sizes of batch,
nr of epochs of data,
stopping monitor - stop training if it does not improve for nr consecutive epochs
"""
batch_size = 128
epochs = 50
early_stopping_monitor = EarlyStopping(patience=2)

hist = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[early_stopping_monitor],
                verbose=1,
                validation_data=(x_validation, y_validation))

"""Save trained model to file"""
model.save('road_signs_no_early_pooling.h5')

"""Print params of NN, visualizes the model"""
print(model.summary())
"""need to install Grphiz, and set env path"""
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""Evaluate the model with the test data to get the scores on "real" data."""
score = model.evaluate(x_validation, y_validation, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

"""Plot results"""
epoch_list = list(range(1, len(hist.history['acc']) + 1)) # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()