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
"""Functions for augmenting a ``tf.data.Dataset``."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.platform import tf_logging as logging

import numpy as np
import tensorflow_addons as tfa


def remove_images_without_objects(X, y, min_objects=1):
    invalid_batches = []
    # Remove images with small numbers of cells
    for b in range(X.shape[0]):
        if len(np.unique(y[b])) - 1 < min_objects:
            invalid_batches.append(b)

    invalid_batches = np.array(invalid_batches, dtype='int')

    if invalid_batches.size > 0:
        logging.warning('Removing %s of %s images with fewer than %s '
                        'objects.', invalid_batches.size, X.shape[0],
                        min_objects)

    X = np.delete(X, invalid_batches, axis=0)
    y = np.delete(y, invalid_batches, axis=0)
    return X, y


def transform_matrix_offset_center(matrix, x, y):
    o_x = tf.constant(float(x) / 2 + 0.5, shape=(1,))
    o_y = tf.constant(float(y) / 2 + 0.5, shape=(1,))
    one = tf.constant(1.0, shape=(1,))
    zero = tf.constant(0.0, shape=(1,))
    offset_row_0 = tf.stack([one, zero, o_x], axis=1)
    offset_row_1 = tf.stack([zero, one, o_y], axis=1)
    offset_row_2 = tf.stack([zero, zero, one], axis=1)
    offset_matrix = tf.concat([offset_row_0, offset_row_1, offset_row_2], axis=0)

    reset_row_0 = tf.stack([one, zero, o_x], axis=1)
    reset_row_1 = tf.stack([zero, one, o_y], axis=1)
    reset_row_2 = tf.stack([zero, zero, one], axis=1)
    reset_matrix = tf.concat([reset_row_0, reset_row_1, reset_row_2], axis=0)
    # offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]], dtype='float32')
    # reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]], dtype='float32')
    # offset_matrix = tf.convert_to_tensor(offset_matrix)
    # reset_matrix = tf.convert_to_tensor(reset_matrix)
    transform_matrix = K.dot(K.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def random_rotate(arr, rotation_range=0):
    """Randomly rotate centroids and appearances.

    Args:
        arr (Tensor): Dictionary of feature data.
        rotation_range (int): Maximum rotation range in degrees.
    Returns:
        Tensor: Rotated ``arr`` data.
    """
    # Calculate the random rotation in radians
    rg = rotation_range * math.pi / 180
    theta = tf.random.uniform(shape=[1], minval=-rg, maxval=rg)

    # Infer interpolation based on input data
    interpolation = 'NEAREST'
    if arr.dtype.isinteger and bool(tf.shape(arr)[-1] > 1):
        interpolation = 'BILINEAR'

    # Transform appearances
    old_shape = tf.shape(arr)
    new_shape = [-1, old_shape[2], old_shape[3], old_shape[4]]
    img = tf.reshape(arr, new_shape)
    img = tfa.image.rotate(img, theta, interpolation=interpolation)
    img = tf.reshape(img, old_shape)
    return img


def random_zoom(arr, zoom_range):
    one = tf.constant(1.0, shape=(1,))
    zero = tf.constant(0.0, shape=(1,))
    zx = tf.random.uniform(shape=(1,), minval=zoom_range[0], maxval=zoom_range[1])
    zy = tf.random.uniform(shape=(1,), minval=zoom_range[0], maxval=zoom_range[1])
    z_row_0 = tf.stack([zx, zero, zero], axis=1)
    z_row_1 = tf.stack([zero, zy, zero], axis=1)
    z_row_2 = tf.stack([zero, zero, one], axis=1)
    zoom_matrix = tf.concat([z_row_0, z_row_1, z_row_2], axis=0)
    h, w = arr.shape[1], arr.shape[2]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)

    # Infer interpolation based on input data
    interpolation = 'NEAREST'
    if arr.dtype.isinteger and bool(tf.shape(arr)[-1] > 1):
        interpolation = 'BILINEAR'

    return tfa.image.transform(arr, transform_matrix, interpolation=interpolation)


def apply_random_transform(*images, rotation_range=0, zoom_range=0,
                           vertical_flip=0, horizontal_flip=0,
                           crop_size=None):
    """Randomly transform the image input array."""
    if crop_size is not None:
        crop_size = conv_utils.normalize_tuple(crop_size, 2, 'crop_size')
        images = tf.image.random_crop(images, crop_size)

    if horizontal_flip:
        images = tf.image.random_flip_left_right(images)

    if vertical_flip:
        images = tf.image.random_flip_up_down(images)

    if rotation_range:
        images = random_rotate(images, rotation_range)

    if zoom_range:
        images = random_zoom(images, zoom_range=zoom_range)

    return images
