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
from skimage.morphology import disk, binary_dilation
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical

try:
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils

from .io_utils import get_image
from .io_utils import get_image_sizes
from .io_utils import nikon_getfiles
from .io_utils import get_immediate_subdirs
from .misc_utils import sorted_nicely
from .plot_utils import plot_training_data_2d
from .plot_utils import plot_training_data_3d
from .transform_utils import distance_transform_2d
from .transform_utils import distance_transform_3d


def get_data(file_name, mode='sample', test_size=.1, seed=None):
    """Load data from NPZ file and split into train and test sets
    # Arguments
        file_name: path to NPZ file to load
        mode: if 'sample', will return datapoints for each pixel,
              otherwise, returns the same data that was loaded
        test_size: percent of data to leave as testing holdout
        seed: seed number for random train/test split repeatability
    # Returns
        dict of training data, and a dict of testing data:
        train_dict, test_dict
    """
    training_data = np.load(file_name)
    X = training_data['X']
    y = training_data['y']
    # win_x = training_data['win_x']
    # win_y = training_data['win_y']

    class_weights = training_data['class_weights'] if 'class_weights' in training_data else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    train_dict = {
        'X': X_train,
        'y': y_train,
        # 'win_x': win_x,
        # 'win_y': win_y,
        'class_weights': class_weights
    }

    test_dict = {
        'X': X_test,
        'y': y_test,
        # 'win_x': win_x,
        # 'win_y': win_y,
        'class_weights': class_weights
    }

    # if X.ndim == 5:
    #     train_dict['win_z'] = training_data['win_z']
    #     test_dict['win_z'] = training_data['win_z']

    return train_dict, test_dict


def get_max_sample_num_list(y, edge_feature, output_mode='sample', padding='valid',
                            window_size_x=30, window_size_y=30):
    """For each set of images and each feature, find the maximum number
    of samples for to be used. This will be used to balance class sampling.
    # Arguments
        y: mask to indicate which pixels belong to which class
        edge_feature: [1, 0, 0], the 1 indicates the feature is the cell edge
        output_mode:  'sample' or 'conv'
        padding:  'valid' or 'same'
    # Returns
        list_of_max_sample_numbers: list of maximum sample size for all classes
    """
    list_of_max_sample_numbers = []

    if padding == 'valid':
        y = trim_padding(y, window_size_x, window_size_y)

    # for each set of images
    for j in range(y.shape[0]):
        if output_mode == 'sample':
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


def sample_label_matrix(y, window_size=(30, 30), padding='valid',
                        max_training_examples=1e7, data_format=None):
    """Create a list of the maximum pixels to sample
    from each feature in each data set.
    """
    data_format = conv_utils.normalize_data_format(data_format)
    is_channels_first = data_format == 'channels_first'
    if is_channels_first:
        num_dirs, num_features, image_size_x, image_size_y = y.shape
    else:
        num_dirs, image_size_x, image_size_y, num_features = y.shape

    window_size = conv_utils.normalize_tuple(window_size, 2, 'window_size')
    window_size_x, window_size_y = window_size

    feature_rows, feature_cols, feature_batch, feature_label = [], [], [], []

    for direc in range(num_dirs):
        for k in range(num_features):
            if is_channels_first:
                feature_rows_temp, feature_cols_temp = np.where(y[direc, k, :, :] == 1)
            else:
                feature_rows_temp, feature_cols_temp = np.where(y[direc, :, :, k] == 1)

            # Check to make sure the features are actually present
            if not feature_rows_temp.size > 0:
                continue

            # Randomly permute index vector
            non_rand_ind = np.arange(len(feature_rows_temp))
            rand_ind = np.random.choice(non_rand_ind, size=len(feature_rows_temp), replace=False)

            for i in rand_ind:
                condition = padding == 'valid' and \
                    feature_rows_temp[i] - window_size_x > 0 and \
                    feature_rows_temp[i] + window_size_x < image_size_x and \
                    feature_cols_temp[i] - window_size_y > 0 and \
                    feature_cols_temp[i] + window_size_y < image_size_y

                if padding == 'same' or condition:
                    feature_rows.append(feature_rows_temp[i])
                    feature_cols.append(feature_cols_temp[i])
                    feature_batch.append(direc)
                    feature_label.append(k)

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


def sample_label_movie(y, window_size=(30, 30, 5), padding='valid',
                       max_training_examples=1e7, data_format=None):
    """Create a list of the maximum pixels to sample from each feature in each
    data set. If output_mode is 'sample', then this will be set to the number
    of edge pixels. If not, it will be set to np.Inf, i.e. sampling everything.
    """
    data_format = conv_utils.normalize_data_format(data_format)
    is_channels_first = data_format == 'channels_first'
    if is_channels_first:
        num_dirs, num_features, image_size_z, image_size_x, image_size_y = y.shape
    else:
        num_dirs, image_size_z, image_size_x, image_size_y, num_features = y.shape

    window_size = conv_utils.normalize_tuple(window_size, 3, 'window_size')
    window_size_x, window_size_y, window_size_z = window_size

    feature_rows, feature_cols, feature_frames, feature_batch, feature_label = [], [], [], [], []

    for d in range(num_dirs):
        for k in range(num_features):
            if is_channels_first:
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
                condition = padding == 'valid' and \
                    frames_temp[i] - window_size_z > 0 and \
                    frames_temp[i] + window_size_z < image_size_z and \
                    rows_temp[i] - window_size_x > 0 and \
                    rows_temp[i] + window_size_x < image_size_x and \
                    cols_temp[i] - window_size_y > 0 and \
                    cols_temp[i] + window_size_y < image_size_y

                if padding == 'same' or condition:
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
        raise ValueError('Expected to trim numpy array of ndim 4 or 5, '
                         'got "{}"'.format(nparr.ndim))
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
    is_channels_first = K.image_data_format() == 'channels_first'
    # Unpack size tuples
    image_size_x, image_size_y = image_size

    # Initialize training data array
    if K.image_data_format() == 'channels_first':
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

                if is_channels_first:
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
    is_channels_first = K.image_data_format() == 'channels_first'
    # Unpack size tuple
    image_size_x, image_size_y = image_size

    # Initialize feature mask array
    if is_channels_first:
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

                if is_channels_first:
                    y[b, l, :, :] = image_data
                else:
                    y[b, :, :, l] = image_data

        # Thin the augmented edges by subtracting the interior features.
        for l, edge in enumerate(edge_feature):
            if edge != 1:
                continue

            for k, non_edge in enumerate(edge_feature):
                if non_edge == 0:
                    if is_channels_first:
                        y[b, l, :, :] -= y[b, k, :, :]
                    else:
                        y[b, :, :, l] -= y[b, :, :, k]

            if is_channels_first:
                y[b, l, :, :] = y[b, l, :, :] > 0
            else:
                y[b, :, :, l] = y[b, :, :, l] > 0

        # Compute the mask for the background
        if is_channels_first:
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
                          window_size=(30, 30),
                          edge_feature=[1, 0, 0],
                          dilation_radius=1,
                          reshape_size=None,
                          padding='valid',
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
        reshape_size: If provided, will reshape the images to the given size
        padding:  'valid' or 'same'
        output_mode:  'sample' or 'conv'
    """
    window_size = conv_utils.normalize_tuple(window_size, 2, 'window_size')
    window_size_x, window_size_y = window_size

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

    # Trim the feature mask so that each window does not overlap with the border of the image
    if padding == 'valid' and output_mode != 'sample':
        y = trim_padding(y, window_size_x, window_size_y)

    # Save training data in npz format
    np.savez(file_name_save, X=X, y=y)

    return None


def load_training_images_3d(direc_name, training_direcs, channel_names, raw_image_direc,
                            image_size, num_frames, montage_mode=False):
    """
    Iterate over every image in the training directories and load
    each into a numpy array.
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    image_size_x, image_size_y = image_size

    # flatten list of lists
    X_dirs = [os.path.join(direc_name, t, raw_image_direc) for t in training_direcs]
    if montage_mode:
        X_dirs = [os.path.join(t, p) for t in X_dirs for p in os.listdir(t)]
        X_dirs = sorted_nicely(X_dirs)

    # Initialize training data array
    if is_channels_first:
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

                if is_channels_first:
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
    is_channels_first = K.image_data_format() == 'channels_first'
    image_size_x, image_size_y = image_size

    # wrapping single annotation name in list for consistency
    if not isinstance(annotation_name, list):
        annotation_name = [annotation_name]

    y_dirs = [os.path.join(direc_name, t, annotation_direc) for t in training_direcs]
    if montage_mode:
        y_dirs = [os.path.join(t, p) for t in y_dirs for p in os.listdir(t)]
        y_dirs = sorted_nicely(y_dirs)

    if is_channels_first:
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
                if is_channels_first:
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
                          window_size=(30, 30, 5),
                          padding='same',
                          output_mode='conv',
                          reshape_size=None,
                          num_frames=50,
                          montage_mode=True):
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
        padding:  'valid' or 'same'
        output_mode:  'sample' or 'conv'
        reshape_size: If provided, will reshape the images to the given size.
        num_frames: number of frames to load from each training directory
        montage_mode: data is broken into several "montage"
                      sub-directories for easier annoation
    """
    window_size = conv_utils.normalize_tuple(window_size, 3, 'window_size')
    window_size_x, window_size_y, window_size_z = window_size

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
    if padding == 'valid' and output_mode != 'sample':
        y = trim_padding(y, window_size_x, window_size_y, window_size_z)

    # Reshape X and y
    if reshape_size is not None:
        X, y = reshape_movie(X, y, reshape_size=reshape_size)

    np.savez(file_name_save, X=X, y=y)

    return None


def make_training_data(direc_name,
                       file_name_save,
                       channel_names,
                       dimensionality,
                       training_direcs=None,
                       window_size=(30, 30),
                       edge_feature=[1, 0, 0],
                       padding='same',
                       output_mode='conv',
                       raw_image_direc='raw',
                       annotation_direc='annotated',
                       reshape_size=None,
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

    padding = conv_utils.normalize_padding(padding)

    if output_mode not in {'sample', 'conv'}:
        raise ValueError('output_mode should be set to either sample or conv')

    if not isinstance(channel_names, list):
        raise ValueError('channel_names should be a list of strings (e.g. [\'DAPI\']). '
                         'Found {}'.format(type(channel_names).__name__))

    if training_direcs is None:
        training_direcs = get_immediate_subdirs(direc_name)

    dimensionality = int(dimensionality)

    if dimensionality == 2:
        make_training_data_2d(direc_name, file_name_save, channel_names,
                              training_direcs=training_direcs,
                              window_size=window_size,
                              edge_feature=edge_feature,
                              reshape_size=reshape_size,
                              padding=padding,
                              output_mode=output_mode,
                              raw_image_direc=raw_image_direc,
                              annotation_direc=annotation_direc)

    elif dimensionality == 3:
        make_training_data_3d(direc_name, file_name_save, channel_names,
                              training_direcs=training_direcs,
                              annotation_name=kwargs.get('annotation_name', 'corrected'),
                              raw_image_direc=raw_image_direc,
                              annotation_direc=annotation_direc,
                              window_size=window_size,
                              padding=padding,
                              output_mode=output_mode,
                              reshape_size=reshape_size,
                              montage_mode=kwargs.get('montage_mode', False),
                              num_frames=kwargs.get('num_frames', 50))

    else:
        raise NotImplementedError('make_training_data is not implemented for '
                                  'dimensionality {}'.format(dimensionality))

    return None
