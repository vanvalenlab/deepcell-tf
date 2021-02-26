# Copyright 2016-2021 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for making training data"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils

from deepcell.utils.tracking_utils import load_trks


def get_data(file_name, mode='sample', test_size=.2, seed=0):
    """Load data from NPZ file and split into train and test sets

    Args:
        file_name (str): path to NPZ file to load
        mode (str): if 'siamese_daughters', returns lineage information from
            .trk file otherwise, returns the same data that was loaded.
        test_size (float): percent of data to leave as testing holdout
        seed (int): seed number for random train/test split repeatability

    Returns:
        (dict, dict): dict of training data, and a dict of testing data
    """
    # siamese_daughters mode is used to import lineage data
    # and associate it with the appropriate batch
    if mode == 'siamese_daughters':
        training_data = load_trks(file_name)
        X = training_data['X']
        y = training_data['y']
        # `daughters` is of the form:
        #
        #                   2 children / cell (potentially empty)
        #                          ___________|__________
        #                         /                      \
        #      daughers = [{id_1: [daughter_1, daughter_2], ...}, ]
        #                  \___________________________________/
        #                                    |
        #                       dict of (cell_id -> children)
        #
        # each batch has a separate (cell_id -> children) dict
        daughters = [{cell: fields['daughters']
                      for cell, fields in tracks.items()}
                     for tracks in training_data['lineages']]

        X_train, X_test, y_train, y_test, ln_train, ln_test = train_test_split(
            X, y, daughters, test_size=test_size, random_state=seed)

        train_dict = {
            'X': X_train,
            'y': y_train,
            'daughters': ln_train
        }

        test_dict = {
            'X': X_test,
            'y': y_test,
            'daughters': ln_test
        }
        return train_dict, test_dict

    training_data = np.load(file_name)
    X = training_data['X']
    y = training_data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    train_dict = {
        'X': X_train,
        'y': y_train
    }

    test_dict = {
        'X': X_test,
        'y': y_test
    }

    return train_dict, test_dict


def get_max_sample_num_list(y, edge_feature, output_mode='sample', padding='valid',
                            window_size_x=30, window_size_y=30):
    """For each set of images and each feature, find the maximum number
    of samples for to be used. This will be used to balance class sampling.

    Args:
        y (numpy.array): mask to indicate which pixels belong to which class
        edge_feature (list): [1, 0, 0], the 1 indicates the feature
            is the cell edge
        output_mode (str):  'sample' or 'conv'
        padding (str): 'valid' or 'same'

    Returns:
        list: list of maximum sample size for all classes
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
    """Sample a 4D Tensor, creating many small images of shape window_size.

    Args:
        y (numpy.array): label masks with the same shape as X data
        window_size (tuple): size of window around each pixel to sample
        padding (str): padding type 'valid' or 'same'
        max_training_examples (int): max number of samples per class
        data_format (str): 'channels_first' or 'channels_last'

    Returns:
        tuple: 4 arrays of coordinates of each sampled pixel
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
    """Sample a 5D Tensor, creating many small voxels of shape window_size.

    Args:
        y (numpy.array): label masks with the same shape as X data
        window_size (tuple): size of window around each pixel to sample
        padding (str): padding type 'valid' or 'same'
        max_training_examples (int): max number of samples per class
        data_format (str): 'channels_first' or 'channels_last'

    Returns:
        tuple: 5 arrays of coordinates of each sampled pixel
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

    Args:
        nparr (numpy.array): numpy array to trim
        win_x (int): number of row pixels to ignore on either side
        win_y (int): number of column pixels to ignore on either side
        win_y (int): number of column pixels to ignore on either side

    Returns:
        numpy.array: trimmed numpy array of size
        ``x - 2 * win_x - 1, y - 2 * win_y - 1``

    Raises:
        ValueError: nparr.ndim is not 4 or 5
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
    E.g. ``reshape_size`` of 256 yields
    (1, 1024, 1024, 1) -> (16, 256, 256, 1)
    The input image is divided into subimages of side length reshape_size,
    with the last row and column of subimages overlapping the one before the
    last if the original image side lengths are not divisible by
    ``reshape_size``.

    Args:
        X (numpy.array): raw 4D image tensor
        y (numpy.array): label mask of 4D image data
        reshape_size (int, list): size of the output tensor
            If input is int, output images are square with side length equal
            reshape_size. If it is a list of 2 ints, then the output images
            size is reshape_size[0] x reshape_size[1]

    Returns:
        numpy.array: reshaped ``X`` and ``y`` 4D tensors
        in ``shape[1:3] = (reshape_size, reshape_size)``,
        if ``reshape_size`` is an ``int``,
        and ``shape[1:3] = reshape_size``,
        if ``reshape_size`` is a list of length 2

    Raises:
        ValueError: ``X.ndim`` is not 4
        ValueError: ``y.ndim`` is not 4
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    if X.ndim != 4:
        raise ValueError('reshape_matrix expects X dim to be 4, got', X.ndim)
    elif y.ndim != 4:
        raise ValueError('reshape_matrix expects y dim to be 4, got', y.ndim)

    if isinstance(reshape_size, int):
        reshape_size_x = reshape_size_y = reshape_size
    elif len(reshape_size) == 2 and all(isinstance(x, int) for x in reshape_size):
        reshape_size_x, reshape_size_y = reshape_size
    else:
        raise ValueError('reshape_size must be an integer or an iterable containing 2 integers.')

    image_size_x, image_size_y = X.shape[2:] if is_channels_first else X.shape[1:3]
    rep_number_x = np.int(np.ceil(np.float(image_size_x) / np.float(reshape_size_x)))
    rep_number_y = np.int(np.ceil(np.float(image_size_y) / np.float(reshape_size_y)))
    new_batch_size = X.shape[0] * rep_number_x * rep_number_y

    if is_channels_first:
        new_X_shape = (new_batch_size, X.shape[1], reshape_size_x, reshape_size_y)
        new_y_shape = (new_batch_size, y.shape[1], reshape_size_x, reshape_size_y)
    else:
        new_X_shape = (new_batch_size, reshape_size_x, reshape_size_y, X.shape[3])
        new_y_shape = (new_batch_size, reshape_size_x, reshape_size_y, y.shape[3])

    new_X = np.zeros(new_X_shape, dtype=K.floatx())
    new_y = np.zeros(new_y_shape, dtype='int32')

    counter = 0
    for b in range(X.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                _axis = 2 if is_channels_first else 1
                if i != rep_number_x - 1:
                    x_start, x_end = i * reshape_size_x, (i + 1) * reshape_size_x
                else:
                    x_start, x_end = -reshape_size_x, X.shape[_axis]

                if j != rep_number_y - 1:
                    y_start, y_end = j * reshape_size_y, (j + 1) * reshape_size_y
                else:
                    y_start, y_end = -reshape_size_y, y.shape[_axis + 1]

                if is_channels_first:
                    new_X[counter] = X[b, :, x_start:x_end, y_start:y_end]
                    new_y[counter] = y[b, :, x_start:x_end, y_start:y_end]
                else:
                    new_X[counter] = X[b, x_start:x_end, y_start:y_end, :]
                    new_y[counter] = y[b, x_start:x_end, y_start:y_end, :]

                new_y[counter] = relabel_movie(new_y[counter])
                counter += 1

    print('Reshaped feature data from {} to {}'.format(y.shape, new_y.shape))
    print('Reshaped training data from {} to {}'.format(X.shape, new_X.shape))
    return new_X, new_y


def relabel_movie(y):
    """Relabels unique instance IDs to be from 1 to N

    Args:
        y (numpy.array): tensor of integer labels

    Returns:
        numpy.array: relabeled tensor with sequential labels
    """
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
    Reshape tensor of dimension 5 to have x and y of size ``reshape_size``.
    Adds overlapping slices to batches.
    E.g. ``reshape_size`` of 256 yields
    ``(1, 5, 1024, 1024, 1) -> (16, 5, 256, 256, 1)``

    Args:
        X (numpy.array): raw 5D image tensor
        y (numpy.array): label mask of 5D image tensor
        reshape_size (int): size of the square output tensor

    Returns:
        numpy.array: reshaped ``X`` and ``y`` tensors in shape
        ``(reshape_size, reshape_size)``

    Raises:
        ValueError: ``X.ndim`` is not 5
        ValueError: ``y.ndim`` is not 5
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
