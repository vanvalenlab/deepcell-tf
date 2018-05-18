"""
dc_data_functions.py

Functions for making training data

@author: David Van Valen
"""

from __future__ import print_function

import fnmatch
import os

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, binary_dilation
from skimage.measure import label
from sklearn.utils import class_weight
from tensorflow.python.keras import backend as K

from .dc_helper_functions import *

"""
Functions to create training data
"""

def get_max_sample_num_list(feature_mask, edge_feature=[1, 0, 0],
                            sample_mode="subsample", border_mode="valid",
                            window_size_x=30, window_size_y=30):
    """
    For each set of images and each feature, find the maximum number of samples
    for to be used.  This will be used to balance class sampling.
    # args
        feature_mask: mask to indicate which pixels belong to which class
        edge_feature: [1, 0, 0], the 1 indicates the feature is the cell edge
        sample_mode:  "subsample" or "all"
        border_mode:  "valid" or "same"
    # returnsf
        list_of_max_sample_numbers: list of maximum sample size for all classes
    """
    feature_mask_trimmed = feature_mask[:, :, window_size_x:-window_size_x, window_size_y:-window_size_y]
    list_of_max_sample_numbers = []
    # for each set of images
    for j in range(feature_mask.shape[0]):
        if sample_mode == "subsample":
            for k, edge_feat in enumerate(edge_feature):
                if edge_feat == 1:
                    if border_mode == "same":
                        list_of_max_sample_numbers += [np.sum(feature_mask[j, k, :, :])]
                    elif border_mode == "valid":
                        list_of_max_sample_numbers += [np.sum(feature_mask_trimmed[j, k, :, :])]

        elif sample_mode == "all":
            list_of_max_sample_numbers += [np.Inf]

    return list_of_max_sample_numbers

def sample_label_matrix(feature_mask, edge_feature, window_size_x=30, window_size_y=30,
                        sample_mode="subsample", border_mode="valid", output_mode="sample"):
    """
    Create a list of the maximum pixels to sample from each feature in each data set.
    If sample_mode is "subsample", then this will be set to the number of edge pixels.
    If not, then it will be set to np.Inf, i.e. sampling everything.
    """
    num_dirs, num_features, image_size_x, image_size_y = feature_mask.shape
    feature_mask_trimmed = feature_mask[:, :, window_size_x:-window_size_x, window_size_y:-window_size_y]

    list_of_max_sample_numbers = get_max_sample_num_list(
        feature_mask=feature_mask, edge_feature=edge_feature,
        window_size_x=window_size_x, window_size_y=window_size_y,
        sample_mode=sample_mode, border_mode=border_mode
    )

    feature_rows = []
    feature_cols = []
    feature_batch = []
    feature_label = []

    if output_mode == "sample":
        for direc in range(num_dirs):
            for k in range(num_features):
                feature_rows_temp, feature_cols_temp = np.where(feature_mask[direc, k, :, :] == 1)

                # Check to make sure the features are actually present
                if feature_rows_temp.size > 0:
                    # Randomly permute index vector
                    non_rand_ind = np.arange(len(feature_rows_temp))
                    rand_ind = np.random.choice(non_rand_ind, size=len(feature_rows_temp), replace=False)

                    for i in rand_ind:
                        if pixel_counter < max_num_of_pixels:
                            if border_mode == "same":
                                condition = True

                            elif border_mode == "valid":
                                condition = feature_rows_temp[i] - window_size_x > 0 and \
                                            feature_rows_temp[i] + window_size_x < image_size_x and \
                                            feature_cols_temp[i] - window_size_y > 0 and \
                                            feature_cols_temp[i] + window_size_y < image_size_y

                            if condition:
                                feature_rows += [feature_rows_temp[i]]
                                feature_cols += [feature_cols_temp[i]]
                                feature_batch += [direc]
                                feature_label += [k]
                                pixel_counter += 1

        # Randomize
        feature_rows = np.array(feature_rows, dtype='int32')
        feature_cols = np.array(feature_cols, dtype='int32')
        feature_batch = np.array(feature_batch, dtype='int32')
        feature_label = np.array(feature_label, dtype='int32')

        non_rand_ind = np.arange(len(feature_rows), dtype='int')
        rand_ind = np.random.choice(non_rand_ind, size=len(feature_rows), replace=False)

        feature_rows = feature_rows[rand_ind]
        feature_cols = feature_cols[rand_ind]
        feature_batch = feature_batch[rand_ind]
        feature_label = feature_label[rand_ind]

        return feature_rows, feature_cols, feature_batch, feature_label

    if output_mode == "conv":
        feature_dict = {}
        if border_mode == "valid":
            feature_mask = feature_mask_trimmed

        for direc in range(feature_mask.shape[0]):
            feature_rows = []
            feature_cols = []
            feature_label = []
            feature_batch = []
            for k in range(feature_mask.shape[1]):
                max_num_of_pixels = list_of_max_sample_numbers[direc]
                pixel_counter = 0

                feature_rows_temp, feature_cols_temp = np.where(feature_mask[direc, k, :, :] == 1)

                # Check to make sure the features are actually present
                if feature_rows_temp.size > 0:
                    # Randomly permute index vector
                    non_rand_ind = np.arange(len(feature_rows_temp))
                    rand_ind = np.random.choice(non_rand_ind, size=len(feature_rows_temp), replace=False)

                    for i in rand_ind:
                        if pixel_counter < max_num_of_pixels:
                            feature_rows += [feature_rows_temp[i]]
                            feature_cols += [feature_cols_temp[i]]
                            feature_batch += [direc]
                            feature_label += [k]
                            pixel_counter += 1

            # Randomize
            non_rand_ind = np.arange(len(feature_rows), dtype='int')
            rand_ind = np.random.choice(non_rand_ind, size=len(feature_rows), replace=False)

            feature_rows = np.array(feature_rows, dtype='int32')
            feature_cols = np.array(feature_cols, dtype='int32')
            feature_batch = np.array(feature_batch, dtype='int32')
            feature_label = np.array(feature_label, dtype='int32')

            feature_rows = feature_rows[rand_ind]
            feature_cols = feature_cols[rand_ind]
            feature_batch = feature_batch[rand_ind]
            feature_label = feature_label[rand_ind]

            feature_dict[direc] = (feature_rows, feature_cols, feature_batch, feature_label)
        return feature_dict

def plot_training_data(channels, feature_mask, max_plotted=5):
    if max_plotted > feature_mask.shape[0]:
        max_plotted = feature_mask.shape[0]

    fig, ax = plt.subplots(max_plotted, feature_mask.shape[1] + 1, squeeze=False)

    for j in range(max_plotted):
        ax[j, 0].imshow(channels[j, 0, :, :], cmap=plt.cm.gray, interpolation='nearest')

        def form_coord(x, y):
            return cf(x, y, channels[j, 0, :, :])

        ax[j, 0].format_coord = form_coord
        ax[j, 0].axes.get_xaxis().set_visible(False)
        ax[j, 0].axes.get_yaxis().set_visible(False)

        for k in range(1, feature_mask.shape[1] + 1):
            ax[j, k].imshow(feature_mask[j, k - 1, :, :], cmap=plt.cm.gray, interpolation='nearest')
            ax[j, k].axes.get_xaxis().set_visible(False)
            ax[j, k].axes.get_yaxis().set_visible(False)
    plt.show()

def reshape_matrix(channels, feature_mask, reshaped_size=256):
    image_size_x, image_size_y = channels.shape[2:]
    rep_number = np.int(np.ceil(np.float(image_size_x) / np.float(reshaped_size)))
    new_batch_size = channels.shape[0] * (rep_number) ** 2

    new_channels = np.zeros((new_batch_size, channels.shape[1], reshaped_size, reshaped_size), dtype=K.floatx())
    new_feature_mask = np.zeros((new_batch_size, feature_mask.shape[1], reshaped_size, reshaped_size), dtype="int32")

    counter = 0
    for batch in range(channels.shape[0]):
        for i in range(rep_number):
            for j in range(rep_number):
                if i != rep_number - 1 and j != rep_number - 1:
                    new_channels[counter, :, :, :] = channels[batch, :, i*reshaped_size:(i+1)*reshaped_size, j*reshaped_size:(j+1)*reshaped_size]
                    new_feature_mask[counter, :, :, :] = feature_mask[batch, :, i*reshaped_size:(i+1)*reshaped_size, j*reshaped_size:(j+1)*reshaped_size]
                if i == rep_number - 1 and j != rep_number - 1:
                    new_channels[counter, :, :, :] = channels[batch, :, -reshaped_size:, j*reshaped_size:(j+1)*reshaped_size]
                    new_feature_mask[counter, :, :, :] = feature_mask[batch, :, -reshaped_size:, j*reshaped_size:(j+1)*reshaped_size]
                if i != rep_number - 1 and j == rep_number - 1:
                    new_channels[counter, :, :, :] = channels[batch, :, i*reshaped_size:(i+1)*reshaped_size, -reshaped_size:]
                    new_feature_mask[counter, :, :, :] = feature_mask[batch, :, i*reshaped_size:(i+1)*reshaped_size, -reshaped_size:]
                if i == rep_number - 1 and j == rep_number - 1:
                    new_channels[counter, :, :, :] = channels[batch, :, -reshaped_size:, -reshaped_size:]
                    new_feature_mask[counter, :, :, :] = feature_mask[batch, :, -reshaped_size:, -reshaped_size:]

                counter += 1
    print('Reshaped feature mask from {} to {}'.format(feature_mask.shape, new_feature_mask.shape))
    print('Reshaped channels from {} to {}'.format(channels.shape, new_channels.shape))
    return new_channels, new_feature_mask

def relabel_movie(feature_mask):
    new_feature_mask = np.zeros(feature_mask.shape)
    unique_cells = list(np.unique(feature_mask))
    relabel_ids = list(np.arange(len(unique_cells)) + 1)
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        cell_loc = np.where(feature_mask == cell_id)
        new_feature_mask[cell_loc] = relabel_id

    return new_feature_mask

def reshape_movie(channels, feature_mask, reshaped_size=256):
    image_size_x, image_size_y = channels.shape[3:]
    rep_number = np.int(np.ceil(np.float(image_size_x)/np.float(reshaped_size)))
    new_batch_size = channels.shape[0] * (rep_number)**2

    new_channels = np.zeros((new_batch_size, channels.shape[1], channels.shape[2], reshaped_size, reshaped_size), dtype=K.floatx())
    new_feature_mask = np.zeros((new_batch_size, feature_mask.shape[1], reshaped_size, reshaped_size), dtype="int32")

    print(new_channels.shape, new_feature_mask.shape)

    counter = 0
    for batch in range(channels.shape[0]):
        for i in range(rep_number):
            for j in range(rep_number):
                if i != rep_number - 1 and j != rep_number - 1:
                    new_channels[counter, :, :,:, :] = channels[batch, :, :, i*reshaped_size:(i+1)*reshaped_size, j*reshaped_size:(j+1)*reshaped_size]
                    feature_mask_temp = relabel_movie(feature_mask[batch, :, i*reshaped_size:(i+1)*reshaped_size, j*reshaped_size:(j+1)*reshaped_size])
                    new_feature_mask[counter, :, :, :] = feature_mask_temp
                if i == rep_number - 1 and j != rep_number - 1:
                    new_channels[counter, :, :, :, :] = channels[batch, :, :, -reshaped_size:, j*reshaped_size:(j+1)*reshaped_size]
                    feature_mask_temp = relabel_movie(feature_mask[batch, :, -reshaped_size:, j*reshaped_size:(j+1)*reshaped_size])
                    new_feature_mask[counter, :, :, :] = feature_mask_temp
                if i != rep_number - 1 and j == rep_number - 1:
                    new_channels[counter, :, :, :, :] = channels[batch, :, :, i*reshaped_size:(i+1)*reshaped_size, -reshaped_size:]
                    feature_mask_temp = relabel_movie(feature_mask[batch, :, i*reshaped_size:(i+1)*reshaped_size, -reshaped_size:])
                    new_feature_mask[counter, :, :, :] = feature_mask_temp
                if i == rep_number - 1 and j == rep_number - 1:
                    new_channels[counter, :, :, :, :] = channels[batch, :, :, -reshaped_size:, -reshaped_size:]
                    feature_mask_temp = relabel_movie(feature_mask[batch, :, -reshaped_size:, -reshaped_size:])
                    new_feature_mask[counter, :, :, :] = feature_mask_temp

                counter += 1

    return new_channels, new_feature_mask

def make_training_data(direc_name, file_name_save, channel_names,
                       training_direcs=None,
                       max_training_examples=1e7,
                       window_size_x=30,
                       window_size_y=30,
                       num_of_features=2,
                       edge_feature=[1, 0, 0],
                       dilation_radius=1,
                       display=False,
                       max_plotted=5,
                       verbose=False,
                       process=True,
                       process_std=False,
                       process_remove_zeros=False,
                       reshape_size=None,
                       border_mode="valid",
                       sample_mode="subsample",
                       output_mode="sample"):
    """
    Read all images in training directories and save as npz file
    # args
        direc_name: directory containing folders of training data
        file_name_save: full filepath for npz file where the data will be saved
        training_direcs: directories of images located inside direc_name.
                         If not provided, all directories in direc_name are used.
        channel_names: List of particular channel name of images to find.
                       channel_name should be in the filename (e.g. "DAPI")
                       If not provided, all images in the training directories
                       without "feature" in their name are used.
        num_of_features: number of classes (e.g. cell interior, cell edge, background)
        edge_feature: List which determines the cell edge feature (usually [1, 0, 0])
                      There can be a single 1 in the list, indicating the index of the feature.
        max_training_examples: max number of samples to be given to model
        window_size_x: number of pixels to +/- x direction to be sampled in sample mode
        window_size_y: number of pixels to +/- y direction to be sampled in sample mode
        dilation_radius: radius for dilating cell edges
        display: whether or not to plot the training data
        max_plotted: how many points to plot if display is True
        verbose:  print more output to screen, similar to DEBUG mode
        process: if True, call process_image on each image
        process_std:  passed to process_image if process is True
        process_remove_zeros:  passed to process_image if process is True
        reshape_size: If provided, will reshape the images to the given size
        border_mode:  "valid" or "same"
        sample_mode:  "subsample" or "all"
        output_mode:  "sample", "conv", or "disc"
    """

    if np.sum(edge_feature) > 1:
        raise ValueError("Only one edge feature is allowed")

    if border_mode not in {"valid", "same"}:
        raise ValueError("border_mode should be set to either valid or same")

    if sample_mode not in {"subsample", "all"}:
        raise ValueError("sample_mode should be set to either subsample or all")

    if output_mode not in {"sample", "conv", "disc"}:
        raise ValueError("output_mode should be set to either sample, conv, or disc")

    if not isinstance(channel_names, list):
        raise ValueError('channel_names should be a list of strings (e.g. ["DAPI"]). '
                         'Found {}'.format(type(channel_names).__name__))

    if training_direcs is None:
        training_direcs = get_immediate_subdirs(direc_name)

    num_direcs = len(training_direcs)
    num_channels = len(channel_names)
    max_training_examples = int(max_training_examples)

    # Load one file to get image sizes
    # All images should be same size bc they are from the same microscope
    image_size_x, image_size_y = get_image_sizes(os.path.join(direc_name, training_direcs[0]), channel_names)

    # Initialize arrays for the training images and the feature masks
    channels = np.zeros((num_direcs, num_channels, image_size_x, image_size_y), dtype='float32')
    feature_mask = np.zeros((num_direcs, num_of_features + 1, image_size_x, image_size_y))

    # Load training images
    for direc_counter, direc in enumerate(training_direcs):
        imglist = os.listdir(os.path.join(direc_name, direc))
        channel_counter = 0

        # Load channels
        for channel_counter, channel in enumerate(channel_names):
            for img in imglist:
                # Regex for matching channel in filename?
                if fnmatch.fnmatch(img, r'*' + channel + r'*'):
                    channel_file = os.path.join(direc_name, direc, img)
                    channel_img = np.asarray(get_image(channel_file), dtype=K.floatx())
                    if process:
                        channel_img = process_image(channel_img, window_size_x, window_size_y,
                                                    remove_zeros=process_remove_zeros,
                                                    std=process_std)
                    # Overwrites if multiple images in one set with same channel
                    channels[direc_counter, channel_counter, :, :] = channel_img

        # Load feature mask
        for j in range(num_of_features):
            feature_name = "feature_" + str(j) + r".*"
            for img in imglist:
                if fnmatch.fnmatch(img, feature_name):
                    feature_file = os.path.join(direc_name, direc, img)
                    feature_img = get_image(feature_file)

                    if np.sum(feature_img) > 0:
                        feature_img /= np.amax(feature_img)

                    if edge_feature[j] == 1 and dilation_radius is not None:
                        # thicken cell edges to be more pronounced
                        strel = disk(dilation_radius)
                        feature_img = binary_dilation(feature_img, selem=strel)

                    feature_mask[direc_counter, j, :, :] = feature_img

        # Thin the augmented edges by subtracting the interior features.
        for j in range(num_of_features):
            if edge_feature[j] == 1:
                for k in range(num_of_features):
                    if edge_feature[k] == 0:
                        feature_mask[direc_counter, j, :, :] -= feature_mask[direc_counter, k, :, :]
                feature_mask[direc_counter, j, :, :] = feature_mask[direc_counter, j, :, :] > 0

        # Compute the mask for the background
        feature_mask_sum = np.sum(feature_mask[direc_counter, :, :, :], axis=0)
        feature_mask[direc_counter, num_of_features, :, :] = 1 - feature_mask_sum

    if reshape_size is not None:
        channels, feature_mask = reshape_matrix(channels, feature_mask, reshaped_size=reshape_size)

    # Trim the feature mask so that each window does not overlap with the border of the image
    feature_mask_trimmed = feature_mask[:, :, window_size_x:-window_size_x, window_size_y:-window_size_y]

    # Sample pixels from the label matrix
    if output_mode == "sample":
        feature_rows, feature_cols, feature_batch, feature_label = sample_label_matrix(
            feature_mask, edge_feature, output_mode=output_mode, sample_mode=sample_mode,
            border_mode=border_mode, window_size_x=window_size_x, window_size_y=window_size_y)

        # Compute weights for each class
        weights = class_weight.compute_class_weight('balanced', y=feature_label,
                                                    classes=np.unique(feature_label))

        # Randomly select training points if there are too many
        if len(feature_rows) > max_training_examples:
            non_rand_ind = np.arange(len(feature_rows), dtype='int')
            rand_ind = np.random.choice(non_rand_ind, size=max_training_examples, replace=False)

            feature_rows = feature_rows[rand_ind]
            feature_cols = feature_cols[rand_ind]
            feature_batch = feature_batch[rand_ind]
            feature_label = feature_label[rand_ind]

        # Save training data in npz format
        np.savez(file_name_save, weights=weights, channels=channels, y=feature_label,
                 batch=feature_batch, pixels_x=feature_rows, pixels_y=feature_cols,
                 win_x=window_size_x, win_y=window_size_y)

    if output_mode == "conv":
        # Create mask of sampled pixels
        feature_rows, feature_cols, feature_batch, feature_label = sample_label_matrix(
            feature_mask, edge_feature, output_mode="sample", sample_mode=sample_mode,
            border_mode=border_mode, window_size_x=window_size_x, window_size_y=window_size_y)

        feature_mask_sample = np.zeros(feature_mask.shape, dtype='int32')
        for b, r, c, l in zip(feature_batch, feature_rows, feature_cols, feature_label):
            feature_mask_sample[b, l, r, c] = 1

        # Compute weights for each class
        weights = class_weight.compute_class_weight('balanced', y=feature_label,
                                                    classes=np.unique(feature_label))

        if border_mode == "valid":
            feature_mask = feature_mask_trimmed

        # Save training data in npz format
        np.savez(file_name_save, class_weights=weights, channels=channels, y=feature_mask,
                 y_sample=feature_mask_sample, win_x=window_size_x, win_y=window_size_y)

    if output_mode == "disc":

        if feature_mask.shape[1] > 3:
            raise ValueError("Only one interior feature is allowed for disc output mode")

        feature_rows, feature_cols, feature_batch, feature_label = sample_label_matrix(feature_mask, edge_feature, output_mode="sample",
                        sample_mode="all", border_mode=border_mode,
                        window_size_x=window_size_x, window_size_y=window_size_y)

        # Compute weights for each class
        weights = class_weight.compute_class_weight('balanced', y=feature_label,
                                                    classes=np.unique(feature_label))

        # Create mask with labeled cells
        feature_mask_label = np.zeros((feature_mask.shape[0], 1, feature_mask.shape[2], feature_mask.shape[3]), dtype='int32')
        for b in range(feature_mask.shape[0]):
            interior_mask = feature_mask[b, 1, :, :]
            feature_mask_label[b, :, :, :] = label(interior_mask)

        max_cells = np.amax(feature_mask_label)
        feature_mask_binary = np.zeros((feature_mask.shape[0], max_cells+1, feature_mask.shape[2], feature_mask.shape[3]), dtype='int32')
        for b in range(feature_mask.shape[0]):
            label_mask = feature_mask_label[b, :, :, :]
            for l in range(max_cells + 1):
                feature_mask_binary[b, l, :, :] = label_mask == l

        if border_mode == "valid":
            feature_mask_label_trimmed = feature_mask_label[:, :, window_size_x:-window_size_x, window_size_y:-window_size_y]
            feature_mask_label = feature_mask_label_trimmed

            feature_mask_binary_trimmed = feature_mask_binary[:, :, window_size_x:-window_size_x, window_size_y:-window_size_y]
            feature_mask_binary = feature_mask_binary_trimmed

        # Save training data in npz format
        np.savez(file_name_save, class_weights=weights, channels=channels,
                 y=feature_mask_binary, win_x=window_size_x, win_y=window_size_y)

    if verbose:
        print("Number of features: %s" % str(feature_mask.shape[1]))
        print("Number of training data points: %s" % str(len(feature_label)))
        print("Class weights: %s" % str(weights))

        for j in range(feature_mask.shape[0]):
            print(1.0/3.0* np.sum(feature_mask[j, :, :, :].astype(K.floatx()), axis=(0, 1, 2)) / np.sum(feature_mask[j, :, :, :].astype(K.floatx()), axis=(1, 2)))

    if display:
        if output_mode == "conv":
            plot_training_data(channels, feature_mask_sample, max_plotted=max_plotted)
        elif output_mode == "disc":
            plot_training_data(channels, feature_mask_label, max_plotted=max_plotted)
        else:
            plot_training_data(channels, feature_mask, max_plotted=max_plotted)

    return None

def make_training_data_movie(direc_name, file_name_save, channel_names,
                             training_direcs=None,
                             annotation_name="corrected",
                             raw_image_direc="RawImages",
                             annotation_direc="Annotation",
                             window_size_x=30,
                             window_size_y=30,
                             border_mode="same",
                             output_mode="disc",
                             reshaped_size=None,
                             process=True,
                             num_frames=50,
                             display=True,
                             num_of_frames_to_display=5,
                             verbose=True):
    """
    Read all images in training directories and save as npz file.
    # args
        direc_name: directory containing folders of training data
        file_name_save: full filepath for npz file where the data will be saved
        training_direcs: directories of images located inside direc_name
                         If not provided, all directories in direc_name are used.
        channel_names: List of particular channel name of images to find.
                       channel_name should be in the filename (e.g. "DAPI")
        annotation_direc: name of folder with annotated images
        raw_image_direc:  name of folder with raw images
        window_size_x: number of pixels to +/- x direction to be sampled in sample mode
        window_size_y: number of pixels to +/- y direction to be sampled in sample mode
        border_mode:  "valid" or "same"
        output_mode:  "sample", "conv", or "disc"
        reshaped_size: If provided, will reshape the images to the given size.
        num_of_features: number of classes (e.g. cell interior, cell edge, background)
        edge_feature: List which determines the cell edge feature (usually [1, 0, 0])
                   There can be a single 1 in the list, indicating the index of the feature.
        max_training_examples: max number of samples to be given to model
        dilation_radius:
        verbose:  print more output to screen, similar to DEBUG mode.
        process:  if True, calls process_image on each image
        num_frames:
        sub_sample: whether or not to subsamble the training data
        display: whether or not to plot the training data
        num_of_frames_to_display:
    """

    if border_mode not in {"valid", "same"}:
        raise ValueError("border_mode should be set to either valid or same")

    if output_mode not in {"sample", "conv", "disc"}:
        raise ValueError("output_mode should be set to either sample, conv, or disc")

    if not isinstance(channel_names, list):
        raise ValueError('channel_names should be a list of strings (e.g. ["DAPI"]). '
                         'Found {}'.format(type(channel_names).__name__))

    if training_direcs is None:
        training_direcs = get_immediate_subdirs(direc_name)

    num_direcs = len(training_direcs)
    num_channels = len(channel_names)

    # Load one file to get image sizes
    image_size_x, image_size_y = get_image_sizes(os.path.join(direc_name, training_direcs[0], raw_image_direc), channel_names)

    # Initialize arrays for the training images and the feature masks
    channels = np.zeros((num_direcs, num_channels, num_frames, image_size_x, image_size_y), dtype='float32')
    feature_label = np.zeros((num_direcs, num_frames, image_size_x, image_size_y))

    # Load training images
    for direc_counter, direc in enumerate(training_direcs):
        # Load channels
        for channel_counter, channel in enumerate(channel_names):
            print(channel)
            print(os.path.join(direc_name, direc, raw_image_direc))
            imglist = nikon_getfiles(os.path.join(direc_name, direc, raw_image_direc), channel)

            for frame_counter, img in enumerate(imglist):
                channel_file = os.path.join(direc_name, direc, raw_image_direc, img)
                channel_img = get_image(channel_file)

                print(np.sum(channel_img.flatten()))

                if process:
                    channel_img = process_image(channel_img, window_size_x, window_size_y)
                channels[direc_counter, channel_counter, frame_counter, :, :] = channel_img

    # Load annotations
    for direc_counter, direc in enumerate(training_direcs):
        imglist = nikon_getfiles(os.path.join(direc_name, direc, annotation_direc), annotation_name)
        for frame_counter, img in enumerate(imglist):
            annotation_file = os.path.join(direc_name, direc, annotation_direc, img)
            annotation_img = get_image(annotation_file)
            feature_label[direc_counter, frame_counter, :, :] = annotation_img

    # Trim annotation images
    if border_mode == "valid":
        feature_label = feature_label[:, :, window_size_x:-window_size_x, window_size_y:-window_size_y]

    # Reshape channels and feature_label
    if reshaped_size is not None:
        channels, feature_label = reshape_movie(channels, feature_label, reshaped_size=reshaped_size)

    # Convert training data to format compatible with discriminative loss function
    if output_mode == "disc":
        max_cells = np.int(np.amax(feature_label))
        feature_mask_binary = np.zeros((feature_label.shape[0], max_cells+1, feature_label.shape[1], feature_label.shape[2], feature_label.shape[3]), dtype='int32')
        for b in range(feature_label.shape[0]):
            label_mask = feature_label[b, :, :, :]
            for l in range(max_cells + 1):
                feature_mask_binary[b, l, :, :, :] = label_mask == l

    # Save training data in npz format
    if output_mode == "disc":
        feature_label = feature_mask_binary

    np.savez(file_name_save, channels=channels, y=feature_label,
             win_x=window_size_x, win_y=window_size_y)

    if display:
        fig, ax = plt.subplots(len(training_direcs), num_of_frames_to_display + 1, squeeze=False)
        print(ax.shape)
        for j in range(len(training_direcs)):
            ax[j, 0].imshow(channels[j, 0, :, :], cmap=plt.cm.gray, interpolation='nearest')

            def form_coord(x, y):
                return cf(x, y, channels[j, 0, :, :])

            ax[j, 0].format_coord = form_coord
            ax[j, 0].axes.get_xaxis().set_visible(False)
            ax[j, 0].axes.get_yaxis().set_visible(False)

            for i in range(num_of_frames_to_display):
                ax[j, i + 1].imshow(feature_label[j, i, :, :], cmap=plt.cm.gray, interpolation='nearest')
                ax[j, i + 1].axes.get_xaxis().set_visible(False)
                ax[j, i + 1].axes.get_yaxis().set_visible(False)
        plt.show()

    if verbose:
        print("Number of cells: %s" % str(max_cells))

    return None
