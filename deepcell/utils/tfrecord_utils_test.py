# Copyright 2016-2022 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
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
# ============================================================================
"""Tests for tfrecord utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import os
import pytest

from deepcell.utils import tfrecord_utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias

    img = np.expand_dims(img, axis=-1)

    return img.astype('float32')


def test__bytes_feature():
    img = _get_image()
    feature = tfrecord_utils._bytes_feature(tf.io.serialize_tensor(img))
    assert hasattr(feature, 'bytes_list')


def test__float_feature():
    value = np.random.rand(1)[0]
    feature = tfrecord_utils._float_feature(value)
    assert hasattr(feature, 'float_list')


def test__int64_feature():
    value = np.random.randint(1)
    feature = tfrecord_utils._int64_feature(value)
    assert hasattr(feature, 'int64_list')


def test_create_segmentation_example():
    X_test = _get_image()
    y_test = _get_image()

    y_test_dict = {'y': y_test}

    example = tfrecord_utils.create_segmentation_example(X_test, y_test_dict)

    for i in range(len(X_test.shape)):
        shape_string = 'X_shape_' + str(i)
        shape = example.features.feature[shape_string].int64_list.value[0]

        assert shape == X_test.shape[i]

    for i in range(len(y_test.shape)):
        shape_string = 'y_shape_' + str(i)
        shape = example.features.feature[shape_string].int64_list.value[0]
        assert shape == y_test.shape[i]


def test_write_segmentation_dataset_to_tfr():
    X_test = _get_image()
    y_test = _get_image()

    X_test = np.expand_dims(X_test, axis=0)
    y_test = np.expand_dims(y_test, axis=0)

    train_dict = {'X': X_test, 'y': y_test}
    filename = 'write_seg_dataset_test'

    tfrecord_utils.write_segmentation_dataset_to_tfr(train_dict,
                                                     filename=filename)

    assert os.path.exists(filename + '.tfrecord')
    assert os.path.exists(filename + '.csv')


def test_parse_segmentation_example():
    X_test = _get_image()
    y_test = _get_image()

    y_test_dict = {'y': y_test}

    example = tfrecord_utils.create_segmentation_example(X_test, y_test_dict)

    dataset_ndims = {'X': 3, 'y': 3}
    parsed = tfrecord_utils.parse_segmentation_example(
        example.SerializeToString(),
        dataset_ndims=dataset_ndims)

    assert np.sum((np.array(parsed[0]['X']) - X_test)**2) == 0
    assert np.sum((np.array(parsed[1]['y']) - y_test)**2) == 0


def test_get_segmentation_dataset():

    X_test = _get_image()
    y_test = _get_image()

    X_test = np.expand_dims(X_test, axis=0)
    y_test = np.expand_dims(y_test, axis=0)

    train_dict = {'X': X_test, 'y': y_test}
    filename = 'write_seg_dataset_test'

    tfrecord_utils.write_segmentation_dataset_to_tfr(train_dict,
                                                     filename=filename,
                                                     verbose=False)

    dataset = tfrecord_utils.get_segmentation_dataset('write_seg_dataset_test')
    it = iter(dataset)

    Xd, yd = it.next()

    X = np.array(Xd['X'])
    y = np.array(yd['y'])

    assert np.sum((X - X_test)**2) == 0
    assert np.sum((y - y_test)**2) == 0
