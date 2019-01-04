import os
os.environ["PATH"] += os.pathsep + 'C:/MyProject/Graphviz2.38/bin/'
from keras.models import load_model
from keras.utils.vis_utils import plot_model

"""Print some prarameters of saved neural network"""
loaded_NN = load_model('road_signs_no_early_pooling.h5')
print(loaded_NN.summary())
print(loaded_NN.get_weights())  # print weights of trained model
print(loaded_NN.optimizer)

"""Print params of NN, visualizes the model"""
print(loaded_NN.summary())
"""need to install Grphiz, and set env path"""
plot_model(loaded_NN, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)
