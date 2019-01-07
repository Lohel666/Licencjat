import os
os.environ["PATH"] += os.pathsep + 'C:/MyProject/Graphviz2.38/bin/'
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import normalize_data as nd

import numpy as np

""" Decodes results from NN to formal output"""


def decode_preditcion(sign_list):
    Decoded_signs = []
    for i in sign_list:
        Decoded_signs.append(set_sign(i))
    return Decoded_signs


"""Definition only for speed limit signs, other recognised signs are categorised as "Inny znak drogowy" """


def set_sign(sign):
    switcher = {
        0: "20 km/h",
        1: "30 km/h",
        2: "50 km/h",
        3: "60 km/h",
        4: "70 km/h",
        5: "80 km/h",
        6: "Koniec ograniczenia prędkości 80 km/h",
        7: "100 km/h",
        8: "120 km/h"
    }
    return switcher.get(sign, "Inny znak drogowy")


""" Collects results from predictions and makes a list"""


def collect_results(prediction):
    predicted_result_list = []
    for p in prediction:
        i, = np.where(p == max(p))
        i = int(i[0])
        predicted_result_list.append(i)
    return predicted_result_list


"""Load NN and print some prarameters of saved neural network"""
loaded_NN = load_model('road_signs_no_early_pooling.h5')
print(loaded_NN.summary())
print(loaded_NN.get_weights())  # print weights of trained model
print(loaded_NN.optimizer)

"""need to install Grphiz, and set env path"""
plot_model(loaded_NN, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)

"""Load some data to catrgorize"""
test_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'
test_images = nd.load_test_data(test_root_path)
normalized_test_images = nd.normalize_images(test_images)
x_test = np.array(normalized_test_images)

"""Run NN to recognise a road sign"""
prediction = loaded_NN.predict(x_test)

result_list = collect_results(prediction)

print("\n".join(decode_preditcion(result_list)))
