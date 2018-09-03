# simple-blobs.py
# Defines a network that can find separate data drom two blobs of data from
# different classes

#imports
from sklearn.datasets import make_blobs
import numpy as np 
import matplotlib.pyplot as plt
os.environ["TFF_CPP_MIN_LOG_"]="2"

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
    amax, bman = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # make prediction with the model and reshape the oputput so contours can plot it
    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8)
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

X, y = make_blobs(n_samples=1000, centers=2, random_state=42)