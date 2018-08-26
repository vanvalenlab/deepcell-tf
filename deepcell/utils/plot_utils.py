"""
plot_utils.py

Functions for plotting data

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tensorflow.python.keras import backend as K


def get_js_video(images, batch=0, channel=0, cmap='jet'):
    """Create a JavaScript video as HTML for visualizing 3D data as a movie"""
    fig = plt.figure()

    ims = []
    for i in range(images.shape[1]):
        im = plt.imshow(images[batch, i, :, :, channel], animated=True, cmap=cmap)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=150, repeat_delay=1000)
    return ani.to_jshtml()


def cf(x_coord, y_coord, sample_image):
    numrows, numcols = sample_image.shape
    col = int(x_coord + 0.5)
    row = int(y_coord + 0.5)
    if 0 <= col < numcols and 0 <= row < numrows:
        z_coord = sample_image[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x_coord, y_coord, z_coord)
    return 'x=%1.4f, y=%1.4f' % (x_coord, y_coord)


def plot_training_data_2d(X, y, max_plotted=5):
    data_format = K.image_data_format()
    if max_plotted > y.shape[0]:
        max_plotted = y.shape[0]

    label_axis = 1 if K.image_data_format() == 'channels_first' else -1

    fig, ax = plt.subplots(max_plotted, y.shape[label_axis] + 1, squeeze=False)

    for i in range(max_plotted):
        X_i = X[i, 0, :, :] if data_format == 'channels_first' else X[i, :, :, 0]
        ax[i, 0].imshow(X_i, cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x_coord, y_coord):
            return cf(x_coord, y_coord, X_i)

        ax[i, 0].format_coord = form_coord
        ax[i, 0].axes.get_xaxis().set_visible(False)
        ax[i, 0].axes.get_yaxis().set_visible(False)

        for j in range(1, y.shape[label_axis] + 1):
            y_k = y[i, j - 1, :, :] if data_format == 'channels_first' else y[i, :, :, j - 1]
            ax[i, j].imshow(y_k, cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[i, j].axes.get_xaxis().set_visible(False)
            ax[i, j].axes.get_yaxis().set_visible(False)

    plt.show()


def plot_training_data_3d(X, y, num_image_stacks, frames_to_display=5):
    data_format = K.image_data_format()
    fig, ax = plt.subplots(num_image_stacks, frames_to_display + 1, squeeze=False)

    for i in range(num_image_stacks):
        X_i = X[i, 0, :, :] if data_format == 'channels_first' else X[i, :, :, 0]
        ax[i, 0].imshow(X_i, cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x_coord, y_coord):
            return cf(x_coord, y_coord, X_i)

        ax[i, 0].format_coord = form_coord
        ax[i, 0].axes.get_xaxis().set_visible(False)
        ax[i, 0].axes.get_yaxis().set_visible(False)

        for j in range(frames_to_display):
            y_j = y[i, j, :, :] if data_format == 'channels_first' else y[i, :, :, j]
            ax[i, j + 1].imshow(y_j, cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[i, j + 1].axes.get_xaxis().set_visible(False)
            ax[i, j + 1].axes.get_yaxis().set_visible(False)
    plt.show()


def plot_error(loss_hist_file, saved_direc, plot_name):
    """Plot the training and validation error from the npz file
    # Arguments
        loss_hist_file: full path to .npz loss history file
        saved_direc: full path to directory where you want to save the plot
        plot_name: the name of plot
    """
    loss_history = np.load(loss_hist_file)
    loss_history = loss_history['loss_history'][()]

    err = np.subtract(1, loss_history['acc'])
    val_err = np.subtract(1, loss_history['val_acc'])

    epoch = np.arange(1, len(err) + 1, 1)
    plt.plot(epoch, err)
    plt.plot(epoch, val_err)
    plt.title('Model Error')
    plt.xlabel('Epoch')
    plt.ylabel('Model Error')
    plt.legend(['Training error', 'Validation error'], loc='upper right')

    filename = os.path.join(saved_direc, plot_name)
    plt.savefig(filename, format='pdf')
