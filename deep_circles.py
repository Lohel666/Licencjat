# deep_circles.py
# Defines a network that can find separate circles of data

#imports
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import os
os.environ["TFF_CPP_MIN_LOG_"]="2"
os.environ["PATH"] += os.pathsep + 'C:/MyProject/Graphviz/bin/'

# helper functions


# plot the data on a figure
def plot_data(pl, X, y):
    # plot class where y==0
    pl.plot(X[y==0,0], X[y==0,1], 'ob', alpha=0.5)
    # plot class where y==1
    pl.plot(X[y==1,0], X[y==1,1], 'xb', alpha=0.5)
    pl.legend(['0', '1'])
    return pl


# common function that draws the decision boundaries
def plot_decision_boundary(model, X, y):

    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make prediction with the model and reshape the oputput so contours can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))
    # plot the contour
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)
    # plot the moons of data
    plot_data(plt, X, y)

    return plt

# Generate some data blobs. Data will be either 0 or 1 when 2 is number of centers.
# X is a [number of samples, 2] sized array. X[sample] contains its x,y position of the sample in the space
# ex: X[1] = [1.342, -2.3], X[2] = [-4.342, 2.12]
# y is a [number of samples] sized array. y[sample] contains the class index (ie. 0 or 1 when there are 2 centers)
# ex: y[1] = 0, y[1] = 1


X, y = make_circles(n_samples=1000, factor=.6, noise=0.1, random_state=42)
pl = plot_data(plt, X, y)
pl.show()

#SPlit the data into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Keras model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# # Simple Sequential model
# model = Sequential()
# model.add(Dense(4, input_shape=(2,), activation="tanh", name="Hidden-1"))
# model.add(Dense(4, activation="tanh", name="Hidden-2"))
# # Add a Dense Fully Connected Layer with 1 neuron. Using input_shape = (2,) says the input will
# #  be arrays of the form (*,2). The first dimension will be unspecified
# #  number of batches (rows) of data. The second dimension is 2 which are the X, Y positions of each data element.
# #  The sigmoid activation function is used to return 0 or 1, signifying the data
# #  cluster the position is predicted to belong to.
# model.add(Dense(1, activation="sigmoid", name="Output_layer"))

#Implement as Functional API 
from keras.models import Model
from keras.layers import Input#
inputs = Input(shape=(2,))
# Hidden layers
x = Dense(4, activation="tanh", name="Hidden-1")(inputs)
x = Dense(4, activation="tanh", name="Hidden-2")(x)
# Output Layer
o = Dense(1, activation="sigmoid", name="Output_layer")(x)
# Create Model and specify the input and output
model = Model(inputs=inputs, outputs=o) 
# Display summary
model.summary()
# Compile the model. Minimize crossentropy for a binary. Maximize for accuracy
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
from keras.utils import plot_model
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
# Define early stopping callback - stops learning process whean NN stops improving
from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor='acc', patience=5, mode=max)]
# Fit the model with the data from make_blobs. Make 100cycles through the data
#  Set verbose to 0 to supress progress messages
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks)
# Get loss and accuracy on test data
eval_result = model.evaluate(X_test, y_test)
# Print test accuracy
print("\n\nTest loss: ", eval_result[0], "Test accuracy: ", eval_result[1])
plot_decision_boundary(model, X, y).show()