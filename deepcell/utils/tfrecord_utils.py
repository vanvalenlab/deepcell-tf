# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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
"""Functions for reading and writing segmentation datasets as TF Records"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.io import serialize_tensor
from tensorflow.keras.utils import to_categorical


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


"""
Functions for segmentation datasets
"""


def create_segmentation_example(X, y_dict):

    # Define the dictionary of our single example
    # Expect label keys in y_dict to be of the form
    # y_*

    data = {}

    data['X'] = _bytes_feature(serialize_tensor(X))

    for i in range(len(X.shape)):
        shape_string = 'X_shape_' + str(i)
        data[shape_string] = _int64_feature(X.shape[i])

    for y in y_dict:
        data[y] = _bytes_feature(serialize_tensor(y_dict[y]))

        for i in range(len(y_dict[y].shape)):
            shape_string = y + '_shape_' + str(i)
            data[shape_string] = _int64_feature(y_dict[y].shape[i])

    # Create an example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=data))

    return example


def write_segmentation_dataset_to_tfr(train_dict,
                                      filename=None,
                                      verbose=True):

    if filename is None:
        ValueError('You need to specify a name for the training dataset!')

    filename_tfrecord = filename + '.tfrecord'
    filename_csv = filename + '.csv'

    count = 0

    # Create the writer that will store the data to disk
    writer = tf.io.TFRecordWriter(filename_tfrecord)

    # Get the images to add
    X = train_dict['X']

    y_keys = []
    for key in train_dict:
        if 'y' in key:
            y_keys.append(key)
    y_list = [train_dict[key] for key in y_keys]

    for b in range(X.shape[0]):
        Xb = X[b]
        yb = {k: y[b] for k, y in zip(y_keys, y_list)}

        example = create_segmentation_example(Xb, yb)

        if example is not None:
            writer.write(example.SerializeToString())
            count += 1

    writer.close()

    # Save dataset metadata
    dataset_keys = train_dict.keys()
    dataset_dims = [len(train_dict[k].shape) - 1 for k in dataset_keys]

    with open(filename_csv, 'w') as f:
        writer = csv.writer(f)
        rows = [[k, dims] for k, dims in zip(dataset_keys, dataset_dims)]
        writer.writerows(rows)

    if verbose:
        print(f'Wrote {count} elements to TFRecord')

    return count


def parse_segmentation_example(example, dataset_ndims=None,
                               X_dtype=tf.float32, y_dtype=tf.float32):

    # Use standard (x,y,c) data structure if not specified
    if dataset_ndims is None:
        dataset_ndims = {'X': 3, 'y': 3}

    # Recreate the example structure
    data = {}

    X_shape_strings = ['X_shape_' + str(i) for i in range(dataset_ndims['X'])]

    y_keys = []
    for key in dataset_ndims:
        if 'y' in key:
            y_keys.append(key)

    y_shape_strings_dict = {}
    for key in y_keys:
        y_shape_strings = [key + '_shape_' + str(i)
                           for i in range(dataset_ndims[key])]
        y_shape_strings_dict[key] = y_shape_strings

    data['X'] = tf.io.FixedLenFeature([], tf.string)
    for ss in X_shape_strings:
        data[ss] = tf.io.FixedLenFeature([], tf.int64)

    for key in y_keys:
        data[key] = tf.io.FixedLenFeature([], tf.string)

        for ss in y_shape_strings_dict[key]:
            data[ss] = tf.io.FixedLenFeature([], tf.int64)

    # Get data
    content = tf.io.parse_single_example(example, data)

    X = content['X']
    y_list = [content[key] for key in y_keys]

    X_shape = [content[ss] for ss in X_shape_strings]
    y_shape_list = []
    for key in y_keys:
        y_shape_list.append([content[ss] for ss in y_shape_strings_dict[key]])

    # Get the feature (our image and labels) and reshape appropriately
    X = tf.io.parse_tensor(X, out_type=X_dtype)
    X = tf.reshape(X, shape=X_shape)
    X_dict = {'X': X}

    y_dict = {}
    for key, y, y_shape in zip(y_keys, y_list, y_shape_list):
        y = tf.io.parse_tensor(y, out_type=y_dtype)
        y = tf.reshape(y, shape=y_shape)

        y_dict[key] = y

    return X_dict, y_dict


"""
Functions for tracking datasets
"""


def sample_batch_from_sparse(sp, batch):
    with tf.device('/cpu:0'):
        shape = sp.shape.as_list()
        n_dim = len(shape)
        start = [batch] + [0] * (n_dim - 1)
        size = [1] + shape[1:]
        sp_slice = tf.sparse.slice(sp, start=start, size=size)
        sp_slice = tf.sparse.reduce_sum(sp_slice, axis=0, keepdims=False,
                                        output_is_sparse=True)
        return sp_slice


def create_sparse_tensor_features(sparse_tensor, name='adj'):
    feature_dict = {}

    val = sparse_tensor.values.numpy()
    ind = sparse_tensor.indices.numpy()
    shape = sparse_tensor.dense_shape.numpy()

    feature_dict[name + '_val'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=val))

    for i in range(ind.shape[-1]):
        feature_dict[name + '_ind_' + str(i)] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=ind[:, i]))
    return feature_dict


def create_tracking_example(track_dict):

    data = {}

    # Define the dictionary of our single example
    for key in track_dict:
        if is_sparse(track_dict[key]):
            ss = create_sparse_tensor_features(track_dict[key], name=key)
            data.update(ss)
        else:
            data[key] = _bytes_feature(serialize_tensor(track_dict[key]))

        shapes = track_dict[key].shape

        for i in range(len(shapes)):
            shape_string = key + '_shape_' + str(i)
            data[shape_string] = _int64_feature(shapes[i])

    # Create an Example, wrapping the single features
    example = tf.train.Example(features=tf.train.Features(feature=data))

    return example


def write_tracking_dataset_to_tfr(track,
                                  target_max_cells=168,
                                  filename=None,
                                  verbose=True):

    filename_tfr = filename + '.tfrecord'
    filename_csv = filename + '.csv'

    count = 0

    writer = tf.io.TFRecordWriter(filename_tfr)

    # Get features to add
    app = track.appearances
    cent = track.centroids
    morph = track.morphologies
    adj = track.norm_adj_matrices
    temp_adj = track.temporal_adj_matrices

    # Pad cells - we need to do this to use validation data
    # during training

    with tf.device('/cpu:0'):
        adj = tf.sparse.to_dense(adj).numpy()
        temp_adj = tf.sparse.to_dense(temp_adj).numpy()

        max_cells = app.shape[2]

        if target_max_cells < max_cells:
            pad_length = 0

        else:
            pad_length = target_max_cells - max_cells
            app = np.pad(app, ((0, 0), (0, 0), (0, pad_length),
                               (0, 0), (0, 0), (0, 0)))
            cent = np.pad(cent, ((0, 0), (0, 0), (0, pad_length), (0, 0)))
            morph = np.pad(morph, ((0, 0), (0, 0), (0, pad_length), (0, 0)))
            adj = np.pad(adj, ((0, 0), (0, 0), (0, pad_length),
                               (0, pad_length)))
            temp_adj = np.pad(temp_adj, ((0, 0), (0, 0), (0, pad_length),
                                         (0, pad_length), (0, 0)))

        adj = track._get_sparse(adj)
        temp_adj = track._get_sparse(temp_adj)

    # Iterate over all batches
    for b in range(app.shape[0]):
        app_b = app[b]
        cent_b = cent[b]
        morph_b = morph[b]
        adj_b = sample_batch_from_sparse(adj, b)
        temp_adj_b = sample_batch_from_sparse(temp_adj, b)

        track_dict = {'app': app_b,
                      'cent': cent_b,
                      'morph': morph_b,
                      'adj': adj_b,
                      'temp_adj': temp_adj_b}

        example = create_tracking_example(track_dict)

        if example is not None:
            writer.write(example.SerializeToString())
            count += 1

    writer.close()

    if verbose:
        print(f'Wrote {count} elements to TFRecord')

    # Save dataset metadata
    dataset_keys = track_dict.keys()
    dataset_dims = [len(track_dict[k].shape) for k in dataset_keys]

    with open(filename_csv, 'w') as f:
        writer = csv.writer(f)
        rows = [[k, dims] for k, dims in zip(dataset_keys, dataset_dims)]
        writer.writerows(rows)

        adj_shape_row = ['adj_shape'] + list(track_dict['adj'].shape)
        writer.writerow(adj_shape_row)

        temp_adj_shape_row = ['temp_adj_shape'] + list(track_dict['temp_adj'].shape)
        writer.writerow(temp_adj_shape_row)

    return count


def parse_tracking_example(example, dataset_ndims,
                           dtype=tf.float32):

    X_names = ['app', 'cent', 'morph', 'adj']
    y_names = ['temp_adj']

    sparse_names = ['adj', 'temp_adj']

    full_name_dict = {'app': 'appearances',
                      'cent': 'centroids',
                      'morph': 'morphologies',
                      'adj': 'adj_matrices',
                      'temp_adj': 'temporal_adj_matrices'}

    # Recreate the example structure
    data = {}
    shape_strings_dict = {}
    shapes_dict = {}

    for key in dataset_ndims:
        if 'shape' in key:
            new_key = '_'.join(key.split('_')[0:-1])
            shapes_dict[new_key] = dataset_ndims[key]

    for key in shapes_dict:
        dataset_ndims.pop(key + '_shape')

    for key in dataset_ndims:
        if key in sparse_names:
            data[key] = tf.io.SparseFeature(value_key=key + '_val',
                                            index_key=[key + '_ind_' + str(i)
                                                       for i in range(dataset_ndims[key])],
                                            size=shapes_dict[key],
                                            dtype=tf.float32)
        else:
            data[key] = tf.io.FixedLenFeature([], tf.string)

        shape_strings = [key + '_shape_' + str(i)
                         for i in range(dataset_ndims[key])]
        shape_strings_dict[key] = shape_strings

        for ss in shape_strings:
            data[ss] = tf.io.FixedLenFeature([], tf.int64)

    # Get data
    content = tf.io.parse_single_example(example, data)

    X_dict = {}
    y_dict = {}

    for key in dataset_ndims:

        # Get the feature and reshape
        if key in sparse_names:
            value = content[key]
        else:
            shape = [content[ss] for ss in shape_strings_dict[key]]
            value = content[key]
            value = tf.io.parse_tensor(value, out_type=dtype)
            value = tf.reshape(value, shape=shape)

        if key in X_names:
            X_dict[full_name_dict[key]] = value
        else:
            y_dict[full_name_dict[key]] = value

    return X_dict, y_dict


def get_dataset(filename, parse_fn=None, **kwargs):

    # Define tfrecord and csv file
    filename_tfrecord = filename + '.tfrecord'
    filename_csv = filename + '.csv'

    # Load the csv
    dataset_ndims = {}
    shapes = {}

    with open(filename_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if 'shape' in row[0]:
                dataset_ndims[row[0]] = [int(i) for i in row[1:]]
            else:
                dataset_ndims[row[0]] = int(row[1])

    # Create the dataset
    dataset = tf.data.TFRecordDataset(filename_tfrecord)

    # Pass each feature through the mapping function
    def parse_func(example):
        return parse_fn(example,
                        dataset_ndims=dataset_ndims,
                        **kwargs)

    dataset = dataset.map(parse_func)

    return dataset


def get_segmentation_dataset(filename, **kwargs):
    return get_dataset(filename, parse_fn=parse_segmentation_example, **kwargs)


def get_tracking_dataset(filename, **kwargs):
    return get_dataset(filename, parse_fn=parse_tracking_example, **kwargs)
