""" Import section """
import os

"""scikit-image is a collection of algorithms for image processing"""
import skimage.data
import skimage.transform

"""Matplotlib is a library used for data visualization"""
import matplotlib
import matplotlib.pyplot as plt
import csv

training_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
training_normalized_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Training_Images/GTSRB/Final_Training/Normalized-Images'
# test_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'

"""
function for load the images obtained from https://hackernoon.com/automatic-recognition-of-speed-limit-signs-deep-learning-with-keras-and-tensorflow-310d90af9826
arguments: path to the traffic sign data,
returns: list of images, list of corresponding labels
"""


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


def load_test_data(root_path):
    images = []
    label_directory = os.path.join(root_path)
    file_names = [os.path.join(label_directory, f)
                  for f in os.listdir(label_directory) if f.endswith(".ppm")]
    for f in file_names:
        images.append(skimage.data.imread(f))
    return images


""" self explanatory - from https://hackernoon.com/automatic-recognition-of-speed-limit-signs-deep-learning-with-keras-and-tensorflow-310d90af9826 """


def show_data_classes(images, labels):
    """Display the first image of each label."""
    miniatures = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in miniatures:
        """ Pick the first image for each label."""
        image = images[labels.index(label)]
        """ A grid of 8 rows x 8 columns"""
        plt.subplot(8, 8, i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


def normalize_images(load_images):
    normalized_images = []
    for image in load_images:
        normalized_images.append(skimage.transform.resize(image, (64, 64)))
    return normalized_images


"""
Following method will print labels with miniature picture that we can verifi our label-image sets
"""
# show_data_classes(load_images, load_labels)
"""
We need to normalize loaded images because neural network hads to got input of the same size.
following loop is used to check what is common size of image in Dataset
"""
"""
# Print some of images to check average sizes
for single_image in load_images[70:90]:
    print("shape: {0}, min: {1}, max: {2}".format(
        single_image.shape, single_image.min(), single_image.max()))

Output:
shape: (34, 34, 3), min: 32, max: 255
shape: (35, 35, 3), min: 22, max: 235
shape: (37, 38, 3), min: 29, max: 235
shape: (39, 39, 3), min: 19, max: 212
shape: (39, 40, 3), min: 25, max: 208
shape: (41, 41, 3), min: 19, max: 218
shape: (44, 44, 3), min: 17, max: 224
shape: (45, 46, 3), min: 18, max: 222
shape: (47, 48, 3), min: 18, max: 248
shape: (50, 51, 3), min: 17, max: 255
shape: (52, 52, 3), min: 9, max: 255
shape: (55, 55, 3), min: 10, max: 255
shape: (59, 59, 3), min: 4, max: 255
shape: (63, 62, 3), min: 8, max: 255
shape: (70, 69, 3), min: 5, max: 255
shape: (75, 74, 3), min: 6, max: 255
shape: (85, 83, 3), min: 3, max: 255
shape: (95, 92, 3), min: 0, max: 255
shape: (108, 107, 3), min: 0, max: 255
shape: (124, 122, 3), min: 0, max: 255
"""
