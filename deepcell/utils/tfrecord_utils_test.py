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
from deepcell.data.tracking import Track
from deepcell.data.tracking_test import get_dummy_data
from deepcell_tracking.utils import get_max_cells


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


"""
Test tracking TFRecord functions
"""


def compare_tracking(output, test_track_dict):

    keys = ['appearances', 'morphologies', 'centroids']
    keys_brev = ['app', 'morph', 'cent']

    for key_0, key_1 in zip(keys, keys_brev):
        assert np.sum((output[0][key_0] - test_track_dict[key_1])**2) == 0

    keys = ['adj_matrices']
    keys_brev = ['adj']

    for key_0, key_1 in zip(keys, keys_brev):
        val_0 = tf.sparse.to_dense(output[0][key_0]).numpy()
        val_1 = tf.sparse.to_dense(test_track_dict[key_1]).numpy()

        assert np.sum((val_0 - val_1)**2) == 0

    keys = ['temporal_adj_matrices']
    keys_brev = ['temp_adj']

    for key_0, key_1 in zip(keys, keys_brev):
        val_0 = tf.sparse.to_dense(output[1][key_0]).numpy()
        val_1 = tf.sparse.to_dense(test_track_dict[key_1]).numpy()

        assert np.sum((val_0 - val_1)**2) == 0


def create_test_track_dict(test_track):

    adj_b = tfrecord_utils.sample_batch_from_sparse(test_track.norm_adj_matrices, 0)
    temp_adj_b = tfrecord_utils.sample_batch_from_sparse(test_track.temporal_adj_matrices, 0)
    test_track_dict = {'app': test_track.appearances[0],
                       'morph': test_track.morphologies[0],
                       'cent': test_track.centroids[0],
                       'adj': adj_b,
                       'temp_adj': temp_adj_b}

    return test_track_dict


def test_sample_batch_from_sparse():
    test_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2])
    test_array = np.expand_dims(test_array, axis=1)

    with tf.device('/cpu:0'):
        sparse_tensor = tf.sparse.from_dense(test_array)
        sliced_1 = tfrecord_utils.sample_batch_from_sparse(sparse_tensor, 3)
        sliced_1 = tf.sparse.to_dense(sliced_1).numpy()

        sliced_2 = tfrecord_utils.sample_batch_from_sparse(sparse_tensor, 9)
        sliced_2 = tf.sparse.to_dense(sliced_2).numpy()

    assert np.unique(sliced_1)[0] == 1
    assert np.unique(sliced_2)[0] == 2


def test_create_sparse_tensor_features():
    test_array = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2])
    test_array = np.expand_dims(test_array, axis=0)

    with tf.device('/cpu:0'):
        sparse_tensor = tf.sparse.from_dense(test_array)

    feature_dict = tfrecord_utils.create_sparse_tensor_features(sparse_tensor,
                                                                name='test')
    val = feature_dict['test_val'].float_list
    ind_0 = feature_dict['test_ind_0'].int64_list
    ind_1 = feature_dict['test_ind_1'].int64_list

    assert val.value == list(sparse_tensor.values.numpy())
    assert ind_0.value == list(sparse_tensor.indices[:, 0].numpy())
    assert ind_1.value == list(sparse_tensor.indices[:, 1].numpy())


def test_create_tracking_example():
    test_dict = get_dummy_data()
    test_track = Track(tracked_data=test_dict)

    test_track_dict = create_test_track_dict(test_track)

    example = tfrecord_utils.create_tracking_example(test_track_dict)

    for key in test_track_dict:
        for i in range(len(test_track_dict[key].shape)):
            shape_string = key + '_shape_' + str(i)
            shape = example.features.feature[shape_string].int64_list.value[0]

            assert shape == test_track_dict[key].shape[i]


def test_write_tracking_dataset_to_tfr():
    test_dict = get_dummy_data()
    test_track = Track(tracked_data=test_dict)
    filename = 'write_track_dataset_test'
    tfrecord_utils.write_tracking_dataset_to_tfr(test_track,
                                                 filename=filename)

    assert os.path.exists(filename + '.tfrecord')
    assert os.path.exists(filename + '.csv')


def test_parse_tracking_example():
    test_dict = get_dummy_data()
    test_track = Track(tracked_data=test_dict)

    test_track_dict = create_test_track_dict(test_track)

    example = tfrecord_utils.create_tracking_example(test_track_dict)
    dataset_ndims = {}

    for key in test_track_dict:
        dataset_ndims[key] = len(test_track_dict[key].shape)

    dataset_ndims['adj_shape'] = list(test_track_dict['adj'].shape)
    dataset_ndims['temp_adj_shape'] = list(test_track_dict['temp_adj'].shape)

    output = tfrecord_utils.parse_tracking_example(example.SerializeToString(),
                                                   dataset_ndims=dataset_ndims)

    compare_tracking(output, test_track_dict)


def test_get_tracking_dataset():
    test_dict = get_dummy_data()
    test_track = Track(tracked_data=test_dict)

    test_track_dict = create_test_track_dict(test_track)

    filename = 'write_track_dataset_test'
    max_cells = get_max_cells(test_dict['y'])
    tfrecord_utils.write_tracking_dataset_to_tfr(test_track,
                                                 filename=filename,
                                                 target_max_cells=max_cells)

    dataset = tfrecord_utils.get_tracking_dataset('write_track_dataset_test')
    it = iter(dataset)

    output = it.next()

    compare_tracking(output, test_track_dict)


def test_get_dataset():
    test_get_segmentation_dataset()
    test_get_tracking_dataset()
