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
"""Prepare datasets for semantic segmentation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deepcell.data import split_dataset
from deepcell.data.augment import apply_random_transform
from deepcell.data.augment import remove_images_without_objects
from deepcell.image_generators import _transform_masks


def transform_labels(y, transforms, transforms_kwargs=None,
                     data_format='channels_last'):
    """Copied from ``deepcell.image_generators.sematnic``"""
    if transforms_kwargs is None:
        transforms_kwargs = {}

    channel_axis = len(y.shape) - 1 if data_format == 'channels_last' else 1

    y_semantic_list = []

    # loop over channels axis of labels in case there are multiple label types
    slc = [slice(None)] * len(y.shape)
    for c in range(y.shape[channel_axis]):
        slc[channel_axis] = slice(c, c + 1)

        y_current = y[tuple(slc)]

        for transform in transforms:
            transform_kwargs = transforms_kwargs.get(transform, dict())
            y_transform = _transform_masks(y_current, transform,
                                           data_format=data_format,
                                           **transform_kwargs)
            y_semantic_list.append(y_transform)

    return y_semantic_list


def randomly_transform_images(X, y, **kwargs):
    if not isinstance(y, list):
        y = [y]

    # send as single list to apply the same transform to all inputs
    input_images = [X, *y]

    transformed_y = []

    transformed_X = apply_random_transform(X, **kwargs)

    for _y in y:
        transformed = apply_random_transform(_y, **kwargs)
        transformed_y.append(transformed)

    if len(transformed_y) == 1:
        transformed_y = transformed_y[0]

    return transformed_X, transformed_y


def prepare_data(X, y, batch_size=32, buffer_size=256,
                 seed=None, min_objects=1, val_split=0.2,
                 rotation_range=0, zoom_range=0, crop_size=None,
                 horizontal_flip=False, vertical_flip=False,
                 transforms=(None,), transforms_kwargs=None):
    """Build and prepare the tracking dataset.

    Args:
        X (tensor): A numpy array of the input data.
        y (tensor): A numpy array of labels of the same shape as ``X``.
        batch_size (int): number of examples per batch.
        buffer_size (int): number of samples to buffer.
        seed (int): Random seed.
        track_length (int): Number of frames per example.
        rotation_range (int): Maximum degrees to rotate inputs.
        zoom_range (int): Maximum range to zoom in or out.
        horizontal_flip (bool): Enable random left-right flipping.
        vertical_flip (bool): Enable random up-down flipping.
        val_split (float): Fraciton of data to split into validation set.

    Returns:
        tf.data.Dataset: A ``tf.data.Dataset`` object ready for training.
    """
    if transforms_kwargs is None:
        transforms_kwargs = dict()

    try:
        if isinstance(transforms, str):
            transforms = [transforms]
        transforms = list(transforms)
    except TypeError as err:
        raise TypeError('transforms should be a list, found {}'.format(
            type(transforms).__name__)) from err

    AUTOTUNE = tf.data.AUTOTUNE

    X, y = remove_images_without_objects(X, y, min_objects=min_objects)

    # create the semantic transforms before creating the dataset
    # they require numpy and cannot be done by the Dataset itself
    y_transform = transform_labels(y, transforms, transforms_kwargs)

    dataset = tf.data.Dataset.from_tensor_slices((X, y_transform))

    dataset = dataset.shuffle(buffer_size, seed=seed).repeat()

    # split into train/val before doing any augmentation
    train_data, val_data = split_dataset(dataset, val_split)

    # randomly rotate, flip, & zoom training data
    transform_train_data = lambda X, y: randomly_transform_images(
        X, y,
        crop_size=crop_size,
        rotation_range=rotation_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        zoom_range=zoom_range)

    train_data = train_data.map(transform_train_data, num_parallel_calls=AUTOTUNE)

    # randomly crop the validation data, but no other transforms
    transform_val_data = lambda X, y: randomly_transform_images(
        X, y, crop_size=crop_size)

    val_data = val_data.map(transform_val_data, num_parallel_calls=AUTOTUNE)

    # batch the data
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    # prefetch the data
    train_data = train_data.prefetch(AUTOTUNE)
    val_data = val_data.prefetch(AUTOTUNE)

    return train_data, val_data
