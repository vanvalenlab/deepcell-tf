"""
dc_plotting_functions.py

Functions for plotting data for visual inspection

@author: David Van Valen
"""

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt

def cf(x_coord, y_coord, sample_image):
	numrows, numcols = sample_image.shape
	col = int(x_coord + 0.5)
	row = int(y_coord + 0.5)
	if col >= 0 and col < numcols and row >= 0 and row < numrows:
		z_coord = sample_image[row, col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x_coord, y_coord, z_coord)
	return 'x=%1.4f, y=1.4%f' % (x_coord, y_coord)

def plot_training_data_2d(X, y, max_plotted=5):
    if max_plotted > y.shape[0]:
        max_plotted = y.shape[0]

    fig, ax = plt.subplots(max_plotted, y.shape[1] + 1, squeeze=False)

    for j in range(max_plotted):
        ax[j, 0].imshow(X[j, 0, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x_coord, y_coord):
            return cf(x_coord, y_coord, X[j, 0, :, :])

        ax[j, 0].format_coord = form_coord
        ax[j, 0].axes.get_xaxis().set_visible(False)
        ax[j, 0].axes.get_yaxis().set_visible(False)

        for k in range(1, y.shape[1] + 1):
            ax[j, k].imshow(y[j, k - 1, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[j, k].axes.get_xaxis().set_visible(False)
            ax[j, k].axes.get_yaxis().set_visible(False)
    plt.show()

def plot_training_data_3d(X, y, num_image_stacks, frames_to_display=5):
    fig, ax = plt.subplots(num_image_stacks, frames_to_display + 1, squeeze=False)

    for j in range(num_image_stacks):
        ax[j, 0].imshow(X[j, 0, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x_coord, y_coord):
            return cf(x_coord, y_coord, X[j, 0, :, :])

        ax[j, 0].format_coord = form_coord
        ax[j, 0].axes.get_xaxis().set_visible(False)
        ax[j, 0].axes.get_yaxis().set_visible(False)

        for i in range(frames_to_display):
            ax[j, i + 1].imshow(y[j, i, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[j, i + 1].axes.get_xaxis().set_visible(False)
            ax[j, i + 1].axes.get_yaxis().set_visible(False)
    plt.show()
