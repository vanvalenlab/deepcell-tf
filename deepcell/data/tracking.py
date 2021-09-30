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
"""Dataset Builders"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from deepcell.data import split_dataset


def temporal_slice(X, y, track_length=8):
    """Randomly slice movies and labels with a length of ``track_length``.

    Args:
        X (dict): Dictionary of feature data.
        y (dict): Dictionary of labels.
        track_length (int): Length of temporal slices.

    Returns:
        tuple(dict, dict): Tuple of sliced ``X`` and ``y`` data.
    """
    temporal_adj_matrices = y['temporal_adj_matrices']

    temporal_adj_matrices = tf.dtypes.cast(temporal_adj_matrices, dtype=tf.dtypes.int32)

    # Identify max time, accounting for padding
    # Padding frames have zero value accross all channels - look for these
    tam_reduce_sum = tf.sparse.reduce_sum(temporal_adj_matrices, axis=[1,2,3])
    non_pad_indices = tf.where(tam_reduce_sum != 0)
    max_time = tf.reduce_max(non_pad_indices) - track_length

    max_time = tf.cond(max_time > 0,
                       lambda: tf.dtypes.cast(max_time, dtype=tf.int32),
                       lambda: tf.dtypes.cast(1, dtype=tf.int32))

    t_start = tf.random.uniform(shape=[], minval=0,
                                maxval=max_time,
                                dtype=tf.int32)

    t_end = t_start + track_length

    def slice_sparse(sp, t_start, t_length):
        shape = sp.shape.as_list()
        n_dim = len(shape)
        start = [t_start] + [0] * (n_dim - 1)
        size = [t_length] + shape[1:]
        sp_slice = tf.sparse.slice(sp, start, size)
        return tf.sparse.to_dense(sp_slice)

    for key, data in X.items():
        if isinstance(data, tf.sparse.SparseTensor):
            X[key] = slice_sparse(data, t_start, track_length)
        else:
            X[key] = data[t_start:t_end]

    for key, data in y.items():
        if isinstance(data, tf.sparse.SparseTensor):
            y[key] = slice_sparse(data, t_start, track_length - 1)
        else:
            y[key] = data[t_start:t_end]

    return (X, y)


def random_rotate(X, y, rotation_range=0):
    """Randomly rotate centroids and appearances.

    Args:
        X (dict): Dictionary of feature data.
        rotation_range (int): Maximum rotation range in degrees.

    Returns:
        dict: Rotated ``X`` data.
    """
    appearances = X['appearances']
    centroids = X['centroids']

    # Calculate the random rotation in radians
    rg = rotation_range * math.pi / 180
    theta = tf.random.uniform(shape=[1], minval=-rg, maxval=rg)

    # Transform appearances
    old_shape = tf.shape(appearances)
    new_shape = [-1, old_shape[2], old_shape[3], old_shape[4]]
    img = tf.reshape(appearances, new_shape)
    img = tfa.image.rotate(img, theta)
    img = tf.reshape(img, old_shape)
    X['appearances'] = img

    # Rotate coordinates
    cos_theta = tf.math.cos(theta)
    sin_theta = tf.math.sin(theta)
    rotation_matrix = tf.concat([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ], axis=1)
    transformed_centroids = tf.matmul(centroids, rotation_matrix)
    X['centroids'] = transformed_centroids

    return X, y


def random_translate(X, y, range=512):
    """Randomly translate the centroids."""
    centroids = X['centroids']
    r0 = tf.random.uniform([1, 1, 2], -range, range)
    transformed_centroids = centroids + r0
    X['centroids'] = transformed_centroids
    return X, y


def prepare_dataset(track_info, batch_size=32, buffer_size=256,
                    seed=None, track_length=8, rotation_range=0,
                    translation_range=0, val_size=0.2, test_size=0):
    """Build and prepare the tracking dataset.

    Args:
        track_info (dict): A dictionary of all input and output features
        batch_size (int): number of examples per batch
        buffer_size (int): number of samples to buffer
        seed (int): Random seed
        track_length (int): Number of frames per example
        rotation_range (int): Maximum degrees to rotate inputs
        translation_range (int): Maximum range of translation,
            should be equivalent to original input image size.
        val_size (float): Fraction of data to split into validation set.
        test_size (float): Fraction of data to split into test set.

    Returns:
        tf.data.Dataset: A ``tf.data.Dataset`` object ready for training.
    """
    input_dict = {
        'appearances': track_info['appearances'],
        'centroids': track_info['centroids'],
        'morphologies': track_info['morphologies'],
        'adj_matrices': track_info['norm_adj_matrices'],
    }

    output_dict = {'temporal_adj_matrices': track_info['temporal_adj_matrices']}

    dataset = tf.data.Dataset.from_tensor_slices((input_dict, output_dict))

    dataset = dataset.shuffle(buffer_size, seed=seed).repeat()

    # randomly sample along the temporal axis
    sample = lambda X, y: temporal_slice(X, y, track_length=track_length)
    dataset = dataset.map(sample, num_parallel_calls=tf.data.AUTOTUNE)

    # split into train/val before doing any augmentation
    train_data, val_data, test_data = split_dataset(dataset, val_size, test_size)

    # randomly rotate
    rotate = lambda X, y: random_rotate(X, y, rotation_range=rotation_range)
    train_data = train_data.map(rotate, num_parallel_calls=tf.data.AUTOTUNE)

    # randomly translate centroids
    translate = lambda X, y: random_translate(X, y, range=translation_range)
    train_data = train_data.map(translate, num_parallel_calls=tf.data.AUTOTUNE)

    # batch the data
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    # prefetch the data
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.AUTOTUNE)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)

    return train_data, val_data, test_data
