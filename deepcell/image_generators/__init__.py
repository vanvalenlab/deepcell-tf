# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""
Custom Image Data Generators used to generate augmented batches of training
data. These custom generators extend the keras.ImageDataGenerator, and allow
for training with label masks, bounding boxes, and more customized annotations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical

from deepcell.utils import transform_utils


def _transform_masks(y, transform, data_format=None, **kwargs):
    """Based on the transform key, apply a transform function to the masks.

    Refer to :mod:`deepcell.utils.transform_utils` for more information about
    available transforms. Caution for unknown transform keys.

    Args:
        y (numpy.array): Labels of ndim 4 or 5
        transform (str): Name of the transform, one of
            {"deepcell", "disc", "watershed", None}
        data_format (str): One of 'channels_first', 'channels_last'.
        kwargs (dict): Optional transform keyword arguments.

    Returns:
        numpy.array: the output of the given transform function on y

    Raises:
        ValueError: Rank of y is not 4 or 5.
        ValueError: Channel dimension of y is not 1.
        ValueError: Transform is invalid value.
    """
    valid_transforms = {
        'deepcell',  # deprecated for "pixelwise"
        'pixelwise',
        'disc',
        'watershed',  # deprecated for "outer-distance"
        'watershed-cont',  # deprecated for "outer-distance"
        'inner-distance',
        'outer-distance',
        'centroid',  # deprecated for "inner-distance"
        'fgbg'
    }
    if data_format is None:
        data_format = K.image_data_format()

    if y.ndim not in {4, 5}:
        raise ValueError('`labels` data must be of ndim 4 or 5.  Got', y.ndim)

    channel_axis = 1 if data_format == 'channels_first' else -1

    if y.shape[channel_axis] != 1:
        raise ValueError('Expected channel axis to be 1 dimension. Got',
                         y.shape[1 if data_format == 'channels_first' else -1])

    if isinstance(transform, str):
        transform = transform.lower()

    if transform not in valid_transforms and transform is not None:
        raise ValueError('`{}` is not a valid transform'.format(transform))

    if transform in {'pixelwise', 'deepcell'}:
        if transform == 'deepcell':
            warnings.warn('The `{}` transform is deprecated. Please use the '
                          '`pixelwise` transform instead.'.format(transform),
                          DeprecationWarning)
        dilation_radius = kwargs.pop('dilation_radius', None)
        separate_edge_classes = kwargs.pop('separate_edge_classes', False)

        edge_class_shape = 4 if separate_edge_classes else 3

        if data_format == 'channels_first':
            y_transform = np.zeros(tuple([y.shape[0]] + [edge_class_shape] + list(y.shape[2:])),
                                   dtype='uint8')
        else:
            y_transform = np.zeros(tuple(list(y.shape[0:-1]) + [edge_class_shape]), dtype='uint8')

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = transform_utils.pixelwise_transform(
                mask, dilation_radius, data_format=data_format,
                separate_edge_classes=separate_edge_classes)

    elif transform in {'outer-distance', 'watershed', 'watershed-cont'}:
        if transform in {'watershed', 'watershed-cont'}:
            warnings.warn('The `{}` transform is deprecated. Please use the '
                          '`outer-distance` transform instead.'.format(transform),
                          DeprecationWarning)

        bins = kwargs.pop('distance_bins', None)
        erosion = kwargs.pop('erosion_width', 0)
        by_frame = kwargs.pop('by_frame', True)

        if data_format == 'channels_first':
            y_transform = np.zeros(tuple([y.shape[0]] + list(y.shape[2:])), dtype='float16')
        else:
            y_transform = np.zeros(y.shape[0:-1], dtype='float16')

        if y.ndim == 5:
            if by_frame:
                _distance_transform = transform_utils.outer_distance_transform_movie
            else:
                _distance_transform = transform_utils.outer_distance_transform_3d
        else:
            _distance_transform = transform_utils.outer_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = _distance_transform(
                mask, bins=bins, erosion_width=erosion)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if bins is None:
            pass
        else:
            # convert to one hot notation
            y_transform = to_categorical(y_transform, num_classes=bins, dtype='uint8')
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform in {'inner-distance', 'centroid'}:
        if transform == 'centroid':
            warnings.warn('The `{}` transform is deprecated. Please use the '
                          '`inner-distance` transform instead.'.format(transform),
                          DeprecationWarning)

        bins = kwargs.pop('distance_bins', None)
        erosion = kwargs.pop('erosion_width', 0)
        by_frame = kwargs.pop('by_frame', True)
        alpha = kwargs.pop('alpha', 0.1)
        beta = kwargs.pop('beta', 1)

        if data_format == 'channels_first':
            y_transform = np.zeros(tuple([y.shape[0]] + list(y.shape[2:])), dtype='float16')
        else:
            y_transform = np.zeros(y.shape[0:-1], dtype='float16')

        if y.ndim == 5:
            if by_frame:
                _distance_transform = transform_utils.inner_distance_transform_movie
            else:
                _distance_transform = transform_utils.inner_distance_transform_3d
        else:
            _distance_transform = transform_utils.inner_distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = _distance_transform(mask, bins=bins,
                                                     erosion_width=erosion,
                                                     alpha=alpha, beta=beta)

        y_transform = np.expand_dims(y_transform, axis=-1)

        if bins is None:
            pass
        else:
            # convert to one hot notation
            y_transform = to_categorical(y_transform, num_classes=bins, dtype='uint8')
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'disc':
        y_transform = to_categorical(y.squeeze(channel_axis), dtype='uint8')
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'fgbg':
        y_transform = np.where(y > 1, 1, y)
        # convert to one hot notation
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, 1, y.ndim)
        y_transform = to_categorical(y_transform, dtype='uint8')
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform is None:
        y_transform = to_categorical(y.squeeze(channel_axis))
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    return y_transform


# Globally-importable utils.
# pylint: disable=wrong-import-position
from deepcell.image_generators.fully_convolutional import ImageFullyConvDataGenerator
from deepcell.image_generators.fully_convolutional import ImageFullyConvIterator
from deepcell.image_generators.fully_convolutional import MovieDataGenerator
from deepcell.image_generators.fully_convolutional import MovieArrayIterator

from deepcell.image_generators.retinanet import RetinaNetGenerator
from deepcell.image_generators.retinanet import RetinaNetIterator
from deepcell.image_generators.retinanet import RetinaMovieIterator
from deepcell.image_generators.retinanet import RetinaMovieDataGenerator

from deepcell.image_generators.semantic import SemanticDataGenerator
from deepcell.image_generators.semantic import SemanticIterator
from deepcell.image_generators.semantic import SemanticMovieGenerator
from deepcell.image_generators.semantic import SemanticMovieIterator

from deepcell.image_generators.sample import SampleDataGenerator
from deepcell.image_generators.sample import ImageSampleArrayIterator
from deepcell.image_generators.sample import SampleMovieDataGenerator
from deepcell.image_generators.sample import SampleMovieArrayIterator

from deepcell.image_generators.scale import ScaleIterator
from deepcell.image_generators.scale import ScaleDataGenerator

from deepcell.image_generators.tracking import SiameseDataGenerator
from deepcell.image_generators.tracking import SiameseIterator

from deepcell.image_generators.cropping import CroppingDataGenerator
from deepcell.image_generators.cropping import CroppingIterator
# pylint: enable=wrong-import-position

del absolute_import
del division
del print_function


__all__ = [
    'ImageFullyConvDataGenerator',
    'ImageFullyConvIterator',
    'MovieDataGenerator',
    'MovieArrayIterator',
    'RetinaNetGenerator',
    'RetinaNetIterator',
    'RetinaMovieIterator',
    'RetinaMovieDataGenerator',
    'SampleDataGenerator',
    'ImageSampleArrayIterator',
    'SampleMovieDataGenerator',
    'SampleMovieArrayIterator',
    'ScaleIterator',
    'ScaleDataGenerator',
    'SiameseDataGenerator',
    'SiameseIterator',
    'CroppingDataGenerator',
    'CroppingIterator'
]
