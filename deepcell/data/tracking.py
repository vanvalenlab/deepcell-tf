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


def temporal_slice(X, y, track_length=8):
    """Randomly slice movies and labels with a length of ``track_length``.

    Args:
        X (dict): Dictionary of feature data.
        y (dict): Dictionary of labels.
        track_length (int): Length of temporal slices.

    Returns:
        tuple(dict, dict): Tuple of sliced ``X`` and ``y`` data.
    """
    # TODO: time axis may change! would be easier if it was always first.
    appearnces = X['appearances']
    max_time = tf.shape(appearnces)[1] - track_length

    t_start = tf.random.uniform(shape=[], minval=0,
                                maxval=max_time,
                                dtype=tf.int32)

    for key in X:
        if 'adj' not in key:
            X[key] = X[key][:, t_start:t_start + track_length, ...]
        else:
            X[key] = X[key][:, :, t_start:t_start + track_length]

    for key in y:
        y[key] = y[key][:, :, t_start:t_start + track_length - 1, ...]

    return (X, y)


def random_rotate(X, rotation_range=0):
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
    # TODO: what does r0 do? why values of 512?
    r0 = tf.random.uniform([1, 1, 2], -512, 512)
    transformed_centroids = transformed_centroids + r0
    X['centroids'] = transformed_centroids

    return X


def prepare_dataset(track_info, batch_size=32, buffer_size=256,
                    seed=None, track_length=8, rotation_range=0):
    """Build and prepare the tracking dataset.

    Args:
        batch_size (int):
        buffer_size (int):
        seed (int):

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

    # TODO: Include a split dataset utils module when ready
    # TODO: the train/test/val split should be addressed

    dataset = tf.data.Dataset.from_tensor_slices((input_dict, output_dict))

    dataset = dataset.shuffle(buffer_size, seed).repeat()

    # randomly sample along the temporal axis
    sample = lambda X, y: temporal_slice(X, y, track_length=track_length)
    dataset = dataset.map(sample)

    # randomly rotate
    rotate = lambda X, y: (random_rotate(X, rotation_range=rotation_range), y)
    dataset = dataset.map(rotate)

    # batch the data
    dataset = dataset.batch(batch_size)

    return dataset
