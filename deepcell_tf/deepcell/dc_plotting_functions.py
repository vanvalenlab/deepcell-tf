"""
dc_plotting_functions.py

Functions for plotting data for visual inspection

@author: David Van Valen
"""

import matplotlib.pyplot as plt

def cf(x, y, sample_image):
	numrows, numcols = sample_image.shape
	col = int(x + 0.5)
	row = int(y + 0.5)
	if col >= 0 and col < numcols and row >= 0 and row < numrows:
		z = sample_image[row, col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
	return 'x=%1.4f, y=1.4%f' % (x, y)

def plot_training_data_2d(channels, feature_mask, max_plotted=5):
    if max_plotted > feature_mask.shape[0]:
        max_plotted = feature_mask.shape[0]

    fig, ax = plt.subplots(max_plotted, feature_mask.shape[1] + 1, squeeze=False)

    for j in range(max_plotted):
        ax[j, 0].imshow(channels[j, 0, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x, y):
            return cf(x, y, channels[j, 0, :, :])

        ax[j, 0].format_coord = form_coord
        ax[j, 0].axes.get_xaxis().set_visible(False)
        ax[j, 0].axes.get_yaxis().set_visible(False)

        for k in range(1, feature_mask.shape[1] + 1):
            ax[j, k].imshow(feature_mask[j, k - 1, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[j, k].axes.get_xaxis().set_visible(False)
            ax[j, k].axes.get_yaxis().set_visible(False)
    plt.show()

def plot_training_data_3d(channels, feature_label, num_image_stacks, frames_to_display=5):
    fig, ax = plt.subplots(num_image_stacks, frames_to_display + 1, squeeze=False)

    for j in range(num_image_stacks):
        ax[j, 0].imshow(channels[j, 0, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')

        def form_coord(x, y):
            return cf(x, y, channels[j, 0, :, :])

        ax[j, 0].format_coord = form_coord
        ax[j, 0].axes.get_xaxis().set_visible(False)
        ax[j, 0].axes.get_yaxis().set_visible(False)

        for i in range(frames_to_display):
            ax[j, i + 1].imshow(feature_label[j, i, :, :], cmap=plt.get_cmap('gray'), interpolation='nearest')
            ax[j, i + 1].axes.get_xaxis().set_visible(False)
            ax[j, i + 1].axes.get_yaxis().set_visible(False)
    plt.show()
