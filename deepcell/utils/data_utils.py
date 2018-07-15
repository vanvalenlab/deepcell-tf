"""
data_utils.py

Functions for making training data

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from fnmatch import fnmatch
import os
import random

import numpy as np
from skimage.measure import label
from skimage.morphology import disk, binary_dilation
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical

from .io_utils import get_image
from .io_utils import get_image_sizes
from .io_utils import nikon_getfiles
from .io_utils import get_immediate_subdirs
from .misc_utils import sorted_nicely
from .plot_utils import plot_training_data_2d
from .plot_utils import plot_training_data_3d
from .transform_utils import distance_transform_2d
from .transform_utils import distance_transform_3d


CHANNELS_FIRST = K.image_data_format() == 'channels_first'


def get_data(file_name, mode='sample', test_size=.1, seed=None):
    """Load data from NPZ file and split into train and test sets
    # Arguments
        file_name: path to NPZ file to load
        mode: if 'sample', will return datapoints for each pixel,
              otherwise, returns the same data that was loaded
        test_size: percent of data to leave as testing holdout
        seed: seed number for random train/test split repeatability
    # Returns
        dict of training data, and a tuple of testing data:
        train_dict, (X_test, y_test)
    """
    training_data = np.load(file_name)
    X = training_data['X']
    y = training_data['y']
    win_x = training_data['win_x']
    win_y = training_data['win_y']
    win_z = None

    class_weights = training_data['class_weights'] if 'class_weights' in training_data else None

    if mode == 'sample' and X.ndim == 4:
        batch = training_data['batch']
        pixels_x = training_data['pixels_x']
        pixels_y = training_data['pixels_y']

        if CHANNELS_FIRST:
            sample_shape = (len(batch), X.shape[1], 2 * win_x + 1, 2 * win_y + 1)
        else:
            sample_shape = (len(batch), 2 * win_x + 1, 2 * win_y + 1, X.shape[3])
        X_sample = np.zeros(sample_shape, dtype=K.floatx())

        for i, (b, px, py) in enumerate(zip(batch, pixels_x, pixels_y)):
            if CHANNELS_FIRST:
                X_sample[i] = X[b, :, px - win_x:px + win_x + 1, py - win_y:py + win_y + 1]
            else:
                X_sample[i] = X[b, px - win_x:px + win_x + 1, py - win_y:py + win_y + 1, :]

        X = X_sample

    elif mode == 'sample' and X.ndim == 5:
        batch = training_data['batch']
        pixels_x = training_data['pixels_x']
        pixels_y = training_data['pixels_y']
        pixels_z = training_data['pixels_z']
        win_z = training_data['win_z']

        if CHANNELS_FIRST:
            sample_shape = (len(batch), X.shape[1], 2 * win_z + 1, 2 * win_x + 1, 2 * win_y + 1)
        else:
            sample_shape = (len(batch), 2 * win_z + 1, 2 * win_x + 1, 2 * win_y + 1, X.shape[4])
        X_sample = np.zeros(sample_shape, dtype=K.floatx())

        for i, (b, px, py, pz) in enumerate(zip(batch, pixels_x, pixels_y, pixels_z)):
            if CHANNELS_FIRST:
                X_sample[i] = X[b, :, pz - win_z:pz + win_z + 1, px - win_x:px + win_x + 1, py - win_y:py + win_y + 1]
            else:
                X_sample[i] = X[b, pz - win_z:pz + win_z + 1, px - win_x:px + win_x + 1, py - win_y:py + win_y + 1, :]

        X = X_sample

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    train_dict = {
        'X': X_train,
        'y': y_train,
        'class_weights': class_weights,
        'win_x': win_x,
        'win_y': win_y
    }

    if win_z:
        train_dict['win_z'] = win_z

    return train_dict, (X_test, y_test)


def get_max_sample_num_list(y, edge_feature, output_mode='sample', border_mode='valid',
                            window_size_x=30, window_size_y=30):
    """For each set of images and each feature, find the maximum number
    of samples for to be used. This will be used to balance class sampling.
    # Arguments
        y: mask to indicate which pixels belong to which class
        edge_feature: [1, 0, 0], the 1 indicates the feature is the cell edge
        output_mode:  'sample', 'conv', or 'disc'
        border_mode:  'valid' or 'same'
    # Returns
        list_of_max_sample_numbers: list of maximum sample size for all classes
    """
    list_of_max_sample_numbers = []

    if border_mode.lower() == 'valid':
        y = trim_padding(y, window_size_x, window_size_y)

    # for each set of images
    for j in range(y.shape[0]):
        if output_mode.lower() == 'sample':
            for k, edge_feat in enumerate(edge_feature):
                if edge_feat == 1:
                    if K.image_data_format() == 'channels_first':
                        y_sum = np.sum(y[j, k, :, :])
                    else:
                        y_sum = np.sum(y[j, :, :, k])
                    list_of_max_sample_numbers.append(y_sum)

        else:
            list_of_max_sample_numbers.append(np.Inf)

    return list_of_max_sample_numbers


def sample_label_matrix(y, edge_feature, window_size_x=30, window_size_y=30,
                        border_mode='valid', output_mode='sample',
                        max_training_examples=1e7):
    """Create a list of the maximum pixels to sample from each feature in each
    data set. If output_mode is 'sample', then this will be set to the number
    of edge pixels. If not, it will be set to np.Inf, i.e. sampling everything.
    """
    is_channel_first = K.image_data_format() == 'channels_first'
    if is_channel_first:
        num_dirs, num_features, image_size_x, image_size_y = y.shape
    else:
        num_dirs, image_size_x, image_size_y, num_features = y.shape

    list_of_max_sample_numbers = get_max_sample_num_list(
        y=y, edge_feature=edge_feature, output_mode=output_mode, border_mode=border_mode,
        window_size_x=window_size_x, window_size_y=window_size_y)

    feature_rows, feature_cols, feature_batch, feature_label = [], [], [], []

    for direc in range(num_dirs):
        for k in range(num_features):
            if is_channel_first:
                feature_rows_temp, feature_cols_temp = np.where(y[direc, k, :, :] == 1)
            else:
                feature_rows_temp, feature_cols_temp = np.where(y[direc, :, :, k] == 1)

            # Check to make sure the features are actually present
            if not feature_rows_temp.size > 0:
                continue

            # Randomly permute index vector
            non_rand_ind = np.arange(len(feature_rows_temp))
            rand_ind = np.random.choice(non_rand_ind, size=len(feature_rows_temp), replace=False)

            pixel_counter = 0
            for i in rand_ind:
                if pixel_counter < list_of_max_sample_numbers[direc]:
                    condition = border_mode == 'valid' and \
                        feature_rows_temp[i] - window_size_x > 0 and \
                        feature_rows_temp[i] + window_size_x < image_size_x and \
                        feature_cols_temp[i] - window_size_y > 0 and \
                        feature_cols_temp[i] + window_size_y < image_size_y

                    if border_mode == 'same' or condition:
                        feature_rows.append(feature_rows_temp[i])
                        feature_cols.append(feature_cols_temp[i])
                        feature_batch.append(direc)
                        feature_label.append(k)
                        pixel_counter += 1

    # Randomize
    non_rand_ind = np.arange(len(feature_rows), dtype='int32')
    if not max_training_examples:
        max_training_examples = non_rand_ind.size
    else:
        max_training_examples = int(max_training_examples)

    limit = min(non_rand_ind.size, max_training_examples)
    rand_ind = np.random.choice(non_rand_ind, size=limit, replace=False)

    feature_rows = np.array(feature_rows, dtype='int32')[rand_ind]
    feature_cols = np.array(feature_cols, dtype='int32')[rand_ind]
    feature_batch = np.array(feature_batch, dtype='int32')[rand_ind]
    feature_label = np.array(feature_label, dtype='int32')[rand_ind]

    return feature_rows, feature_cols, feature_batch, feature_label


def sample_label_movie(y, window_size_x=30, window_size_y=30, window_size_z=5,
                       border_mode='valid', max_training_examples=1e7):
    """Create a list of the maximum pixels to sample from each feature in each
    data set. If output_mode is 'sample', then this will be set to the number
    of edge pixels. If not, it will be set to np.Inf, i.e. sampling everything.
    """
    is_channel_first = K.image_data_format() == 'channels_first'
    if is_channel_first:
        num_dirs, num_features, image_size_z, image_size_x, image_size_y = y.shape
    else:
        num_dirs, image_size_z, image_size_x, image_size_y, num_features = y.shape

    feature_rows, feature_cols, feature_frames, feature_batch, feature_label = [], [], [], [], []

    for d in range(num_dirs):
        for k in range(num_features):
            if is_channel_first:
                frames_temp, rows_temp, cols_temp = np.where(y[d, k, :, :, :] == 1)
            else:
                frames_temp, rows_temp, cols_temp = np.where(y[d, :, :, :, k] == 1)

            # Check to make sure the features are actually present
            if not rows_temp.size > 0:
                continue

            # Randomly permute index vector
            non_rand_ind = np.arange(len(rows_temp))
            rand_ind = np.random.choice(non_rand_ind, size=len(rows_temp), replace=False)

            for i in rand_ind:
                condition = border_mode == 'valid' and \
                    frames_temp[i] - window_size_z > 0 and \
                    frames_temp[i] + window_size_z < image_size_z and \
                    rows_temp[i] - window_size_x > 0 and \
                    rows_temp[i] + window_size_x < image_size_x and \
                    cols_temp[i] - window_size_y > 0 and \
                    cols_temp[i] + window_size_y < image_size_y

                if border_mode == 'same' or condition:
                    feature_rows.append(rows_temp[i])
                    feature_cols.append(cols_temp[i])
                    feature_frames.append(frames_temp[i])
                    feature_batch.append(d)
                    feature_label.append(k)

    # Randomize
    non_rand_ind = np.arange(len(feature_rows), dtype='int32')
    if not max_training_examples:
        max_training_examples = non_rand_ind.size
    else:
        max_training_examples = int(max_training_examples)

    limit = min(non_rand_ind.size, max_training_examples)
    rand_ind = np.random.choice(non_rand_ind, size=limit, replace=False)

    feature_frames = np.array(feature_frames, dtype='int32')[rand_ind]
    feature_rows = np.array(feature_rows, dtype='int32')[rand_ind]
    feature_cols = np.array(feature_cols, dtype='int32')[rand_ind]
    feature_batch = np.array(feature_batch, dtype='int32')[rand_ind]
    feature_label = np.array(feature_label, dtype='int32')[rand_ind]

    return feature_frames, feature_rows, feature_cols, feature_batch, feature_label


def trim_padding(nparr, win_x, win_y, win_z=None):
    """Trim the boundaries of the numpy array to allow for a sliding
    window of size (win_x, win_y) to not slide over regions without pixel data
    Aguments:
        nparr: numpy array to trim
        win_x: number of row pixels to ignore on either side
        win_y: number of column pixels to ignore on either side
    Returns:
        trimmed numpy array of size x - 2 * win_x - 1, y - 2 * win_y - 1
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    if nparr.ndim == 4:
        if is_channels_first:
            trimmed = nparr[:, :, win_x:-win_x, win_y:-win_y]
        else:
            trimmed = nparr[:, win_x:-win_x, win_y:-win_y, :]
    elif nparr.ndim == 5:
        if is_channels_first:
            if win_z:
                win_z = int(win_z)
                trimmed = nparr[:, :, win_z:-win_z, win_x:-win_x, win_y:-win_y]
            else:
                trimmed = nparr[:, :, :, win_x:-win_x, win_y:-win_y]
        else:
            if win_z:
                win_z = int(win_z)
                trimmed = nparr[:, win_z:-win_z, win_x:-win_x, win_y:-win_y, :]
            else:
                trimmed = nparr[:, :, win_x:-win_x, win_y:-win_y, :]
    else:
        raise ValueError('Expected to trim numpy array of ndim 4 or 5, got "{}"'.format(
            nparr.ndim))
    return trimmed


def reshape_matrix(X, y, reshape_size=256):
    """
    Reshape matrix of dimension 4 to have x and y of size reshape_size.
    Adds overlapping slices to batches.
    E.g. reshape_size of 256 yields (1, 1024, 1024, 1) -> (16, 256, 256, 1)
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    if X.ndim != 4:
        raise ValueError('reshape_matrix expects X dim to be 4, got {}'.format(X.ndim))
    elif y.ndim != 4:
        raise ValueError('reshape_matrix expects y dim to be 4, got {}'.format(y.ndim))
    image_size_x, image_size_y = X.shape[2:] if is_channels_first else X.shape[1:3]
    rep_number = np.int(np.ceil(np.float(image_size_x) / np.float(reshape_size)))
    new_batch_size = X.shape[0] * (rep_number) ** 2

    if is_channels_first:
        new_X_shape = (new_batch_size, X.shape[1], reshape_size, reshape_size)
        new_y_shape = (new_batch_size, y.shape[1], reshape_size, reshape_size)
    else:
        new_X_shape = (new_batch_size, reshape_size, reshape_size, X.shape[3])
        new_y_shape = (new_batch_size, reshape_size, reshape_size, y.shape[3])

    new_X = np.zeros(new_X_shape, dtype=K.floatx())
    new_y = np.zeros(new_y_shape, dtype='int32')

    counter = 0
    for b in range(X.shape[0]):
        for i in range(rep_number):
            for j in range(rep_number):
                if i != rep_number - 1:
                    x_start, x_end = i * reshape_size, (i + 1) * reshape_size
                else:
                    x_start, x_end = -reshape_size, X.shape[2 if is_channels_first else 1]

                if j != rep_number - 1:
                    y_start, y_end = j * reshape_size, (j + 1) * reshape_size
                else:
                    y_start, y_end = -reshape_size, y.shape[3 if is_channels_first else 2]

                if is_channels_first:
                    new_X[counter] = X[b, :, x_start:x_end, y_start:y_end]
                    new_y[counter] = y[b, :, x_start:x_end, y_start:y_end]
                else:
                    new_X[counter] = X[b, x_start:x_end, y_start:y_end, :]
                    new_y[counter] = y[b, x_start:x_end, y_start:y_end, :]

                counter += 1

    print('Reshaped feature data from {} to {}'.format(y.shape, new_y.shape))
    print('Reshaped training data from {} to {}'.format(X.shape, new_X.shape))
    return new_X, new_y


def relabel_movie(y):
    """Relabels unique instance IDs to be from 1 to N"""
    new_y = np.zeros(y.shape)
    unique_cells = np.unique(y)  # get all unique values of y
    unique_cells = np.delete(unique_cells, 0)  # remove 0, as it is background
    relabel_ids = np.arange(1, len(unique_cells) + 1)
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        cell_loc = np.where(y == cell_id)
        new_y[cell_loc] = relabel_id
    return new_y


def reshape_movie(X, y, reshape_size=256):
    """
    Reshape tensor of dimension 5 to have x and y of size reshape_size.
    Adds overlapping slices to batches.
    E.g. reshape_size of 256 yields (1, 5, 1024, 1024, 1) -> (16, 5, 256, 256, 1)
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    if X.ndim != 5:
        raise ValueError('reshape_movie expects X dim to be 5, got {}'.format(X.ndim))
    elif y.ndim != 5:
        raise ValueError('reshape_movie expects y dim to be 5, got {}'.format(y.ndim))
    image_size_x, image_size_y = X.shape[3:] if is_channels_first else X.shape[2:4]
    rep_number = np.int(np.ceil(np.float(image_size_x) / np.float(reshape_size)))
    new_batch_size = X.shape[0] * (rep_number) ** 2

    if is_channels_first:
        new_X_shape = (new_batch_size, X.shape[1], X.shape[2], reshape_size, reshape_size)
        new_y_shape = (new_batch_size, y.shape[1], y.shape[2], reshape_size, reshape_size)
    else:
        new_X_shape = (new_batch_size, X.shape[1], reshape_size, reshape_size, X.shape[4])
        new_y_shape = (new_batch_size, y.shape[1], reshape_size, reshape_size, y.shape[4])

    new_X = np.zeros(new_X_shape, dtype=K.floatx())
    new_y = np.zeros(new_y_shape, dtype='int32')

    counter = 0
    row_axis = 3 if is_channels_first else 2
    col_axis = 4 if is_channels_first else 3
    for b in range(X.shape[0]):
        for i in range(rep_number):
            for j in range(rep_number):
                if i != rep_number - 1:
                    x_start, x_end = i * reshape_size, (i + 1) * reshape_size
                else:
                    x_start, x_end = -reshape_size, X.shape[row_axis]
                if j != rep_number - 1:
                    y_start, y_end = j * reshape_size, (j + 1) * reshape_size
                else:
                    y_start, y_end = -reshape_size, y.shape[col_axis]

                if is_channels_first:
                    new_X[counter] = X[b, :, :, x_start:x_end, y_start:y_end]
                    new_y[counter] = relabel_movie(y[b, :, :, x_start:x_end, y_start:y_end])
                else:
                    new_X[counter] = X[b, :, x_start:x_end, y_start:y_end, :]
                    new_y[counter] = relabel_movie(y[b, :, x_start:x_end, y_start:y_end, :])

                counter += 1

    print('Reshaped feature data from {} to {}'.format(y.shape, new_y.shape))
    print('Reshaped training data from {} to {}'.format(X.shape, new_X.shape))
    return new_X, new_y


def load_training_images_2d(direc_name, training_direcs, channel_names, image_size,
                            raw_image_direc):
    """
    Iterate over every image in the training directories and load
    each into a numpy array.
    """
    # Unpack size tuples
    image_size_x, image_size_y = image_size

    # Initialize training data array
    if CHANNELS_FIRST:
        X_shape = (len(training_direcs), len(channel_names), image_size_x, image_size_y)
    else:
        X_shape = (len(training_direcs), image_size_x, image_size_y, len(channel_names))

    X = np.zeros(X_shape, dtype=K.floatx())

    # Load training images
    for b, direc in enumerate(training_direcs):
        # e.g. "/data/ecoli/kc", "set1", "RawImages",
        imglist = os.listdir(os.path.join(direc_name, direc, raw_image_direc))

        for c, channel in enumerate(channel_names):
            for img in imglist:
                # if channel string is NOT in image file name, skip it.
                if not fnmatch(img, '*{}*'.format(channel)):
                    continue

                image_file = os.path.join(direc_name, direc, raw_image_direc, img)
                image_data = np.asarray(get_image(image_file), dtype=K.floatx())

                if CHANNELS_FIRST:
                    X[b, c, :, :] = image_data
                else:
                    X[b, :, :, c] = image_data

    return X


def load_annotated_images_2d(direc_name, training_direcs, image_size, edge_feature,
                             dilation_radius, annotation_direc):
    """
    Iterate over every annotated image in the training directories and load
    each into a numpy array.
    """
    # Unpack size tuple
    image_size_x, image_size_y = image_size

    # Initialize feature mask array
    if CHANNELS_FIRST:
        y_shape = (len(training_direcs), len(edge_feature), image_size_x, image_size_y)
    else:
        y_shape = (len(training_direcs), image_size_x, image_size_y, len(edge_feature))

    y = np.zeros(y_shape, dtype='int32')

    for b, direc in enumerate(training_direcs):
        imglist = os.listdir(os.path.join(direc_name, direc, annotation_direc))

        for l, edge in enumerate(edge_feature):
            for img in imglist:
                # if feature string is NOT in image file name, skip it.
                if not fnmatch(img, '*feature_{}*'.format(l)):
                    continue

                image_data = get_image(os.path.join(direc_name, direc, annotation_direc, img))

                if np.sum(image_data) > 0:
                    image_data /= np.amax(image_data)

                if edge == 1 and dilation_radius is not None:
                    # thicken cell edges to be more pronounced
                    image_data = binary_dilation(image_data, selem=disk(dilation_radius))

                if CHANNELS_FIRST:
                    y[b, l, :, :] = image_data
                else:
                    y[b, :, :, l] = image_data

        # Thin the augmented edges by subtracting the interior features.
        for l, edge in enumerate(edge_feature):
            if edge != 1:
                continue

            for k, non_edge in enumerate(edge_feature):
                if non_edge == 0:
                    if CHANNELS_FIRST:
                        y[b, l, :, :] -= y[b, k, :, :]
                    else:
                        y[b, :, :, l] -= y[b, :, :, k]

            if CHANNELS_FIRST:
                y[b, l, :, :] = y[b, l, :, :] > 0
            else:
                y[b, :, :, l] = y[b, :, :, l] > 0

        # Compute the mask for the background
        if CHANNELS_FIRST:
            y[b, len(edge_feature) - 1, :, :] = 1 - np.sum(y[b], axis=0)
        else:
            y[b, :, :, len(edge_feature) - 1] = 1 - np.sum(y[b], axis=2)

    return y


def make_training_data_2d(direc_name,
                          file_name_save,
                          channel_names,
                          raw_image_direc='raw',
                          annotation_direc='annotated',
                          training_direcs=None,
                          max_training_examples=1e7,
                          distance_transform=False,
                          distance_bins=4,
                          window_size_x=30,
                          window_size_y=30,
                          edge_feature=[1, 0, 0],
                          dilation_radius=1,
                          display=False,
                          max_plotted=5,
                          verbose=False,
                          reshape_size=None,
                          border_mode='valid',
                          output_mode='sample'):
    """
    Read all images in training directories and save as npz file
    # Arguments
        direc_name: directory containing folders of training data
        file_name_save: full filepath for npz file where the data will be saved
        training_direcs: directories of images located inside direc_name.
                         If not provided, all directories in direc_name are used.
        channel_names: List of particular channel name of images to find.
                       channel_name should be in the filename (e.g. 'DAPI')
                       If not provided, all images in the training directories
                       without 'feature' in their name are used.
        edge_feature: List which determines the cell edge feature (usually [1, 0, 0])
                      There can be a single 1 in the list, indicating the index of the feature.
        max_training_examples: max number of samples to be given to model
        window_size_x: number of pixels to +/- x direction to be sampled in sample mode
        window_size_y: number of pixels to +/- y direction to be sampled in sample mode
        dilation_radius: radius for dilating cell edges
        display: whether or not to plot the training data
        max_plotted: how many points to plot if display is True
        verbose:  print more output to screen, similar to DEBUG mode
        reshape_size: If provided, will reshape the images to the given size
        border_mode:  'valid' or 'same'
        output_mode:  'sample', 'conv', or 'disc'
    """
    # Load one file to get image sizes (all images same size as they are from same microscope)
    image_path = os.path.join(direc_name, random.choice(training_direcs), raw_image_direc)
    image_size = get_image_sizes(image_path, channel_names)

    X = load_training_images_2d(direc_name, training_direcs, channel_names,
                                image_size=image_size,
                                raw_image_direc=raw_image_direc)

    y = load_annotated_images_2d(direc_name, training_direcs,
                                 image_size=image_size,
                                 edge_feature=edge_feature,
                                 annotation_direc=annotation_direc,
                                 dilation_radius=dilation_radius)

    if reshape_size is not None:
        X, y = reshape_matrix(X, y, reshape_size=reshape_size)

    if distance_transform:
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
            new_y = np.zeros((y.shape[0], 1, y.shape[2], y.shape[3]), dtype='int32')
        else:
            new_y = np.zeros((y.shape[0], y.shape[1], y.shape[2], 1), dtype='int32')
            channel_axis = -1
        for b in range(y.shape[0]):
            if K.image_data_format() == 'channels_first':
                d = np.expand_dims(y[b, 1, :, :], axis=channel_axis)
            else:
                d = np.expand_dims(y[b, :, :, 1], axis=channel_axis)
            new_y[b] = distance_transform_2d(d, bins=distance_bins)
        y = to_categorical(new_y)
        # not really edge_feature anymore, but there will be the fewest
        # "center" pixels, so lets call that the edge_feature for now
        edge_feature = [0] * y.shape[channel_axis]
        edge_feature[-1] = 1

    # Create mask of sampled pixels
    feature_rows, feature_cols, feature_batch, feature_label = sample_label_matrix(
        y, edge_feature, output_mode=output_mode, border_mode=border_mode,
        window_size_x=window_size_x, window_size_y=window_size_y,
        max_training_examples=max_training_examples)

    weights = compute_class_weight('balanced', y=feature_label, classes=np.unique(feature_label))

    # Sample pixels from the label matrix
    if output_mode == 'sample':

        # Save training data in npz format
        np.savez(file_name_save, class_weights=weights, X=X, y=feature_label,
                 batch=feature_batch, pixels_x=feature_rows, pixels_y=feature_cols,
                 win_x=window_size_x, win_y=window_size_y)

    elif output_mode == 'conv':
        y_sample = np.zeros(y.shape, dtype='int32')
        for b, r, c, l in zip(feature_batch, feature_rows, feature_cols, feature_label):
            if CHANNELS_FIRST:
                y_sample[b, l, r, c] = 1
            else:
                y_sample[b, r, c, l] = 1

        # Trim the feature mask so that each window does not overlap with the border of the image
        if border_mode == 'valid':
            y = trim_padding(y, window_size_x, window_size_y)

        # Save training data in npz format
        np.savez(file_name_save, class_weights=weights, X=X, y=y,
                 y_sample=y_sample, win_x=window_size_x, win_y=window_size_y)

    elif output_mode == 'disc':
        if y.shape[1 if CHANNELS_FIRST else -1] > 3:
            raise ValueError('Only one interior feature is allowed for disc output mode')

        # Create mask with labeled cells
        if CHANNELS_FIRST:
            y_label = np.zeros((y.shape[0], 1, y.shape[2], y.shape[3]), dtype='int32')
        else:
            y_label = np.zeros((y.shape[0], y.shape[1], y.shape[2], 1), dtype='int32')

        for b in range(y.shape[0]):
            if CHANNELS_FIRST:
                interior_mask = y[b, 1, :, :]
                y_label[b, 0, :, :] = label(interior_mask)
            else:
                interior_mask = y[b, :, :, 1]
                y_label[b, :, :, 0] = label(interior_mask)

        max_cells = np.amax(y_label) + 1
        if CHANNELS_FIRST:
            y_binary = np.zeros((y.shape[0], max_cells, y.shape[2], y.shape[3]), dtype='int32')
        else:
            y_binary = np.zeros((y.shape[0], y.shape[1], y.shape[2], max_cells), dtype='int32')

        for b in range(y.shape[0]):
            label_mask = y_label[b]
            for l in range(max_cells):
                if CHANNELS_FIRST:
                    y_binary[b, l, :, :] = label_mask == l
                else:
                    y_binary[b, :, :, l] = label_mask == l

        # Trim the sides of the mask to ensure a sliding window does not slide
        # past before or after the boundary of y_label or y_binary
        if border_mode == 'valid':
            y_label = trim_padding(y_label, window_size_x, window_size_y)
            y_binary = trim_padding(y_binary, window_size_x, window_size_y)

        # Save training data in npz format
        np.savez(file_name_save, class_weights=weights, X=X, y=y_binary,
                 win_x=window_size_x, win_y=window_size_y)

    if verbose:
        print('Number of features: {}'.format(y.shape[1 if CHANNELS_FIRST else -1]))
        print('Number of training data points: {}'.format(len(feature_label)))
        print('Class weights: {}'.format(weights))

    if display:
        if output_mode == 'conv':
            display_mask = y_sample
        elif output_mode == 'disc':
            display_mask = y_label
        else:
            display_mask = y
        plot_training_data_2d(X, display_mask, max_plotted=max_plotted)


def load_training_images_3d(direc_name, training_direcs, channel_names, raw_image_direc,
                            image_size, num_frames, montage_mode=False):
    """
    Iterate over every image in the training directories and load
    each into a numpy array.
    """
    image_size_x, image_size_y = image_size

    # flatten list of lists
    X_dirs = [os.path.join(direc_name, t, raw_image_direc) for t in training_direcs]
    if montage_mode:
        X_dirs = [os.path.join(t, p) for t in X_dirs for p in os.listdir(t)]
        X_dirs = sorted_nicely(X_dirs)

    # Initialize training data array
    if CHANNELS_FIRST:
        X_shape = (len(X_dirs), len(channel_names), num_frames, image_size_x, image_size_y)
    else:
        X_shape = (len(X_dirs), num_frames, image_size_x, image_size_y, len(channel_names))

    X = np.zeros(X_shape, dtype=K.floatx())

    # Load 3D training images
    for b, direc in enumerate(X_dirs):

        for c, channel in enumerate(channel_names):

            imglist = nikon_getfiles(direc, channel)

            for i, img in enumerate(imglist):
                if i >= num_frames:
                    print('Skipped final {skip} frames of {dir}, as num_frames '
                          'is {num} but there are {total} total frames'.format(
                              skip=len(imglist) - num_frames,
                              dir=direc,
                              num=num_frames,
                              total=len(imglist)))
                    break

                image_data = np.asarray(get_image(os.path.join(direc, img)))

                if CHANNELS_FIRST:
                    X[b, c, i, :, :] = image_data
                else:
                    X[b, i, :, :, c] = image_data

    return X


def load_annotated_images_3d(direc_name, training_direcs, annotation_direc, annotation_name,
                             num_frames, image_size, montage_mode=False):
    """
    Iterate over every annotated image in the training directories and load
    each into a numpy array.
    """
    image_size_x, image_size_y = image_size

    # wrapping single annotation name in list for consistency
    if not isinstance(annotation_name, list):
        annotation_name = [annotation_name]

    y_dirs = [os.path.join(direc_name, t, annotation_direc) for t in training_direcs]
    if montage_mode:
        y_dirs = [os.path.join(t, p) for t in y_dirs for p in os.listdir(t)]
        y_dirs = sorted_nicely(y_dirs)

    if CHANNELS_FIRST:
        y_shape = (len(y_dirs), len(annotation_name), num_frames, image_size_x, image_size_y)
    else:
        y_shape = (len(y_dirs), num_frames, image_size_x, image_size_y, len(annotation_name))

    y = np.zeros(y_shape, dtype='int32')

    for b, direc in enumerate(y_dirs):
        for c, name in enumerate(annotation_name):
            imglist = nikon_getfiles(direc, name)

            for z, img_file in enumerate(imglist):
                if z >= num_frames:
                    print('Skipped final {skip} frames of {dir}, as num_frames '
                          'is {num} but there are {total} total frames'.format(
                              skip=len(imglist) - num_frames,
                              dir=direc,
                              num=num_frames,
                              total=len(imglist)))
                    break

                annotation_img = get_image(os.path.join(direc, img_file))
                if CHANNELS_FIRST:
                    y[b, c, z, :, :] = annotation_img
                else:
                    y[b, z, :, :, c] = annotation_img

    return y


def make_training_data_3d(direc_name,
                          file_name_save,
                          channel_names,
                          training_direcs=None,
                          annotation_name='corrected',
                          raw_image_direc='raw',
                          annotation_direc='annotated',
                          window_size_x=30,
                          window_size_y=30,
                          window_size_z=5,
                          border_mode='same',
                          output_mode='disc',
                          reshape_size=None,
                          num_frames=50,
                          display=True,
                          num_of_frames_to_display=5,
                          montage_mode=True,
                          max_training_examples=1e7,
                          distance_transform=False,
                          distance_bins=4,
                          verbose=True):
    """
    Read all images in training directories and save as npz file.
    3D image sets are "stacks" of images.  For annotation purposes, these images
    have been sliced into "montages", where a section of each stack has been sliced
    so they can be efficiently annotated by human users. In this case, the raw_image_direc
    should be a specific montage (e.g. montage_0_0) and the annotation is the corresponding
    annotated montage.  Each montage must maintain the full stack, but can be processed
    independently.
    # Arguments
        direc_name: directory containing folders of training data
        file_name_save: full filepath for npz file where the data will be saved
        training_direcs: directories of images located inside direc_name
                         If not provided, all directories in direc_name are used.
        channel_names: List of particular channel name of images to find.
                       channel_name should be in the filename (e.g. 'DAPI')
        annotation_direc: name of folder with annotated images
        raw_image_direc:  name of folder with raw images
        window_size_x: number of pixels to +/- x direction to be sampled in sample mode
        window_size_y: number of pixels to +/- y direction to be sampled in sample mode
        border_mode:  'valid' or 'same'
        output_mode:  'sample', 'conv', or 'disc'
        reshape_size: If provided, will reshape the images to the given size.
        num_of_features: number of classes (e.g. cell interior, cell edge, background)
        edge_feature: List which determines the cell edge feature (usually [1, 0, 0])
                   There can be a single 1 in the list, indicating the index of the feature.
        max_training_examples: max number of samples to be given to model
        dilation_radius:
        verbose:  print more output to screen, similar to DEBUG mode.
        num_frames:
        sub_sample: whether or not to subsamble the training data
        display: whether or not to plot the training data
        num_of_frames_to_display:
        montage_mode: data is broken into several "montage"
                      sub-directories for easier annoation
    """
    # Load one file to get image sizes
    rand_train_dir = os.path.join(direc_name, random.choice(training_direcs), raw_image_direc)
    if montage_mode:
        rand_train_dir = os.path.join(rand_train_dir, random.choice(os.listdir(rand_train_dir)))

    image_size = get_image_sizes(rand_train_dir, channel_names)

    X = load_training_images_3d(direc_name, training_direcs, channel_names, raw_image_direc,
                                image_size, num_frames=num_frames, montage_mode=montage_mode)

    y = load_annotated_images_3d(direc_name, training_direcs, annotation_direc,
                                 annotation_name, num_frames, image_size,
                                 montage_mode=montage_mode)

    # Trim annotation images
    if border_mode == 'valid':
        y = trim_padding(y, window_size_x, window_size_y, window_size_z)

    # Reshape X and y
    if reshape_size is not None:
        X, y = reshape_movie(X, y, reshape_size=reshape_size)

    if distance_transform:
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
            new_y = np.zeros((y.shape[0], 1, y.shape[2], y.shape[3], y.shape[4]), dtype='int32')
        else:
            new_y = np.zeros((y.shape[0], y.shape[1], y.shape[2], y.shape[3], 1), dtype='int32')
            channel_axis = -1
        for b in range(y.shape[0]):
            d = distance_transform_3d(y[b], bins=distance_bins)
            new_y[b] = np.expand_dims(d, axis=channel_axis)
        y = new_y

    # Sample pixels from the label matrix
    if output_mode == 'sample':
        if not distance_transform:
            y[y > 0] = 1  # make each cell instance equal to 1.
        y = to_categorical(y)

        feat_frames, feat_rows, feat_cols, feat_batch, feat_label = sample_label_movie(
            y=y,
            border_mode=border_mode,
            window_size_x=window_size_x,
            window_size_y=window_size_y,
            window_size_z=window_size_z,
            max_training_examples=max_training_examples)
        # Save training data in npz format
        np.savez(file_name_save, X=X, y=feat_label, batch=feat_batch,
                 pixels_x=feat_rows, pixels_y=feat_cols, pixels_z=feat_frames,
                 win_x=window_size_x, win_y=window_size_y, win_z=window_size_z)

    # Convert training data to format compatible with discriminative loss function
    if output_mode == 'disc':
        max_cells = np.int(np.amax(y))
        if CHANNELS_FIRST:
            binary_mask_shape = (y.shape[0], max_cells + 1, y.shape[1], y.shape[2], y.shape[3])
        else:
            binary_mask_shape = (y.shape[0], y.shape[1], y.shape[2], y.shape[3], max_cells + 1)
        y_binary = np.zeros(binary_mask_shape, dtype='int32')
        for b in range(y.shape[0]):
            label_mask = y[b]
            for l in range(max_cells + 1):
                if CHANNELS_FIRST:
                    y_binary[b, l, :, :, :] = label_mask == l
                else:
                    y_binary[b, :, :, :, l] = label_mask == l

        y = y_binary

        if verbose:
            print('Number of cells: {}'.format(max_cells))

        # Save training data in npz format
        np.savez(file_name_save, X=X, y=y, win_x=window_size_x, win_y=window_size_y, win_z=window_size_z)

    if output_mode == 'conv':
        np.savez(file_name_save, X=X, y=y, win_x=window_size_x, win_y=window_size_y, win_z=window_size_z)

    if display:
        plot_training_data_3d(X, y, len(training_direcs), num_of_frames_to_display)

    return None


def make_training_data(direc_name,
                       file_name_save,
                       channel_names,
                       dimensionality,
                       training_direcs=None,
                       window_size_x=30,
                       window_size_y=30,
                       edge_feature=[1, 0, 0],
                       border_mode='valid',
                       output_mode='sample',
                       raw_image_direc='raw',
                       annotation_direc='annotated',
                       verbose=False,
                       reshape_size=None,
                       display=False,
                       max_training_examples=1e7,
                       distance_transform=False,
                       distance_bins=4,
                       **kwargs):
    """
    Wrapper function for other make_training_data functions (2d, 3d)
    Calls one of the above functions based on the dimensionality of the data
    """
    # Validate Arguments
    if not isinstance(dimensionality, int) and not isinstance(dimensionality, float):
        raise ValueError('Data dimensionality should be an integer value, typically 2 or 3. '
                         'Recieved {}'.format(type(dimensionality).__name__))

    if np.sum(edge_feature) > 1:
        raise ValueError('Only one edge feature is allowed')

    if border_mode not in {'valid', 'same'}:
        raise ValueError('border_mode should be set to either valid or same')

    if output_mode not in {'sample', 'conv', 'disc'}:
        raise ValueError('output_mode should be set to either sample, conv, or disc')

    if not isinstance(channel_names, list):
        raise ValueError('channel_names should be a list of strings (e.g. [\'DAPI\']). '
                         'Found {}'.format(type(channel_names).__name__))

    if training_direcs is None:
        training_direcs = get_immediate_subdirs(direc_name)

    dimensionality = int(dimensionality)

    if dimensionality == 2:
        make_training_data_2d(direc_name, file_name_save, channel_names,
                              training_direcs=training_direcs,
                              window_size_x=window_size_x,
                              window_size_y=window_size_y,
                              edge_feature=edge_feature,
                              distance_transform=distance_transform,
                              distance_bins=distance_bins,
                              display=display,
                              verbose=verbose,
                              reshape_size=reshape_size,
                              border_mode=border_mode,
                              output_mode=output_mode,
                              raw_image_direc=raw_image_direc,
                              annotation_direc=annotation_direc,
                              dilation_radius=kwargs.get('dilation_radius', 1),
                              max_plotted=kwargs.get('max_plotted', 5),
                              max_training_examples=max_training_examples)

    elif dimensionality == 3:
        make_training_data_3d(direc_name, file_name_save, channel_names,
                              training_direcs=training_direcs,
                              annotation_name=kwargs.get('annotation_name', 'corrected'),
                              raw_image_direc=raw_image_direc,
                              annotation_direc=annotation_direc,
                              window_size_x=window_size_x,
                              window_size_y=window_size_y,
                              window_size_z=kwargs.get('window_size_z', 5),
                              distance_transform=distance_transform,
                              distance_bins=distance_bins,
                              border_mode=border_mode,
                              output_mode=output_mode,
                              reshape_size=reshape_size,
                              verbose=verbose,
                              display=display,
                              max_training_examples=max_training_examples,
                              montage_mode=kwargs.get('montage_mode', False),
                              num_frames=kwargs.get('num_frames', 50),
                              num_of_frames_to_display=kwargs.get('num_of_frames_to_display', 5))

    else:
        raise NotImplementedError('make_training_data is not implemented for '
                                  'dimensionality {}'.format(dimensionality))

    return None
