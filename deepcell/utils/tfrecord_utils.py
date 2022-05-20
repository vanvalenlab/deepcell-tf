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
"""deepcell-toolbox Data Utilities"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import numpy as np
import tensorflow as tf

from tensorflow.data import Dataset
from tensorflow.io import serialize_tensor
from tensorflow.keras.utils import to_categorical

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

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
        yb = {k:y[b] for k,y in zip(y_keys, y_list)} 

        example = create_segmentation_example(Xb, yb)

        if example is not None:
            writer.write(example.SerializeToString())
            count += 1

    writer.close()

    # Save dataset metadata
    dataset_keys = train_dict.keys()
    dataset_dims = [len(train_dict[k].shape)-1 for k in dataset_keys]

    with open(filename_csv, 'w') as f:
        writer = csv.writer(f)
        rows = [[k, dims] for k,dims in zip(dataset_keys, dataset_dims)]
        writer.writerows(rows)

    if verbose:
        print(f'Wrote {count} elements to TFRecord')

    return count

def parse_segmentation_example(example, dataset_ndims=None, 
            X_dtype=tf.float32, y_dtype=tf.float32):

    # Use standard (x,y,c) data structure if not specified
    if dataset_ndims is None:
        dataset_ndims = {'X': 3, 'y':3}
        
    # Recreate the example structure
    data = {}

    X_shape_strings = ['X_shape_' + str(i) for i in range(dataset_ndims['X'])]

    y_keys = []
    for key in dataset_ndims:
        if 'y' in key:
            y_keys.append(key)
    
    y_shape_strings_dict = {}
    for key in y_keys:
        y_shape_strings = [key + '_shape_' + str(i) for i in range(dataset_ndims[key])]
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

def get_segmentation_dataset(filename, **kwargs):

    # Define tfrecord and csv file
    filename_tfrecord = filename + '.tfrecord'
    filename_csv = filename + '.csv'

    # Load the csv
    dataset_ndims = {}
    with open(filename_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            dataset_ndims[row[0]] = int(row[1])

    # Create the dataset
    dataset = tf.data.TFRecordDataset(filename_tfrecord)

    # Pass each feature through the mapping function
    def parse_fn(example):
        return parse_segmentation_example(example, **kwargs)

    dataset = dataset.map(parse_fn)

    return dataset







