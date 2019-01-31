import os
os.environ["PATH"] += os.pathsep + 'C:/MyProject/Graphviz2.38/bin/'
from keras.models import load_model
from keras.utils.vis_utils import plot_model

import normalize_data as nd

import numpy as np

""" Decodes results from NN to formal output"""

trained_NN_path = 'C:/MyProject/AI/road_signs_final.h5'
loaded_NN = load_model(trained_NN_path)

def decode_prediction(sign_list):
    decoded_signs_list = []
    for i in sign_list:
        decoded_signs_list.append(set_sign(i))
    return decoded_signs_list


"""
Categorizes int to road sign type
"""


def set_sign(sign):
    switcher = {
        0: "Ograniczenie prędkości 20 km/h",
        1: "Ograniczenie prędkości 30 km/h",
        2: "Ograniczenie prędkości 50 km/h",
        3: "Ograniczenie prędkości 60 km/h",
        4: "Ograniczenie prędkości 70 km/h",
        5: "Ograniczenie prędkości 80 km/h",
        6: "Koniec ograniczenia prędkości 80 km/h",
        7: "Ograniczenie prędkości 100 km/h",
        8: "Ograniczenie prędkości 120 km/h",
        9: "Zakaz wyprzedzania",
        10: "Zakaz wyprzedzania przez samochody ciężarowe",
        11: "Skrzyżowanie z drogą podporządkowaną występującą po obu stronach",
        12: "Droga z pierwszeństwem",
        13: "Ustąp pierwszeństwa",
        14: "STOP",
        15: "Zakaz ruchu w obu kierunkach",
        16: "Zakaz wjazdu samochodów ciężarowych",
        17: "Zakaz wjazdu",
        18: "Inne niebezpieczeństwo",
        19: "Niebezpieczny zakręt w lewo",
        20: "Niebezpieczny zakręt w prawo",
        21: "Niebezpieczne zakręty, pierwszy w lewo",
        22: "Nierówna droga",
        23: "Śliska jezdnia",
        24: "Zwężenie jezdni - prawostronne",
        25: "Roboty drogowe",
        26: "Sygnały świetlne",
        27: "Przejście dla pieszych",
        28: "Dzieci",
        29: "Rowerzyści",
        30: "Oszronienie jezdni",
        31: "Zwierzęta dzikie",
        32: "Koniec zakazów",
        33: "Nakaz jazdy w prawo za znakiem",
        34: "Nakaz jazdy w lewo za znakiem",
        35: "Nakaz jazdy prosto",
        36: "Nakaz jazdy prosto lub w prawo",
        37: "Nakaz jazdy prosto lub w lewo",
        38: "Nakaz jazdy z prawej strony znaku",
        39: "Nakaz jazdy z lewej strony znaku",
        40: "Ruch okrężny",
        41: "Koniec zakazu wyprzedzania",
        42: "Koniec zakazu wyprzedzania przez samochody ciężarowe"
    }
    return switcher.get(sign, "Nie rozpoznany znak drogowy")


""" 
Receives a list of vectors with predictions, and from each vector gets the index of max probablity
(witch will be further convert to road sign type)
"""


def collect_results(prediction):
    predicted_result_list = []
    for p in prediction:
        # in case of equal max probabilities this returns first index of one of the max values
        i, = np.where(p == max(p))
        i = int(i[0])
        predicted_result_list.append(i)
    return predicted_result_list


def load_neural_network(path):
    return load_model(path)


def recognise_road_sign(image_path):
    # this code could be more compact, but less readable
    image = nd.load_single_test_image(image_path) # load image from path to array
    normalized_image = nd.normalize_images(image) # normalize image
    numpy_array = np.array(normalized_image) # load normalized image to numpy array
    probability = loaded_NN.predict(numpy_array) # runs NN and returns vector of probabilities
    prediction = collect_results(probability) # return the type of road sign (int 0-42) 
    decoded_prediction = decode_prediction(prediction) # decode int to road sign description
    return decoded_prediction

def example_of_group_recognition():
    """Load NN and print some prarameters of saved neural network"""
    loaded_NN = load_neural_network(trained_NN_path)
    print(loaded_NN.summary())
    print(loaded_NN.get_weights())  # print weights of trained model
    print(loaded_NN.optimizer)

    """need to install Grphiz, and set env path"""
    plot_model(loaded_NN, to_file='model_plot.png',
            show_shapes=True, show_layer_names=True)

    """Load some data to categorize"""
    test_root_path = 'C:/MyProject/Datasets/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images'
    test_images = nd.load_test_data(test_root_path)
    normalized_test_images = nd.normalize_images(test_images)
    x_test = np.array(normalized_test_images)
    """
    Run NN to recognise a road sign
    returns a list of vectors. Every vector contain a probablity that image belongs to one of the signs type
    then from every vector gets the index of the max elemnt (max probability of prediction of specific road sign)
    finally converts the specific index to road sign description
    """
    prediction = loaded_NN.predict(x_test)
    result_list = collect_results(prediction)
    print("\n".join(decode_prediction(result_list)))

example_of_group_recognition()
