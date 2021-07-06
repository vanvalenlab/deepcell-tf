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
"""Tests for tracking functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test

from deepcell.data import tracking


@parameterized.named_parameters([
    {
        'testcase_name': 'test_data_01',
        'time': 30,
        'max_cells': 25,
        'crop_size': 32,
        'batch_size': 32,
        'seed': None,
        'track_length': 8,
        'val_split': 0.4
    }, {
        'testcase_name': 'test_data_02',
        'time': 40,
        'max_cells': 29,
        'crop_size': 32,
        'batch_size': 24,
        'seed': 2,
        'track_length': 7,
        'val_split': 0.2
    }, {
        'testcase_name': 'test_data_03',
        'time': 50,
        'max_cells': 15,
        'crop_size': 32,
        'batch_size': 10,
        'seed': 4,
        'track_length': 6,
        'val_split': 0.1
    }
])
class TrackingTests(test.TestCase, parameterized.TestCase):

    def test_temporal_slice(self, time, max_cells, crop_size,
                            batch_size, seed, track_length, val_split):
        X, y = self.create_test_data(time, max_cells, crop_size)
        sliced_X, sliced_y = tracking.temporal_slice(X, y, track_length)

        # Test temporal dimension correct index and value
        for key in sliced_X:
            self.assertEqual(sliced_X[key].shape[0], track_length)
        for key in sliced_y:
            self.assertEqual(sliced_y[key].shape[0], track_length - 1)

    def test_random_rotate(self, time, max_cells, crop_size,
                           batch_size, seed, track_length, val_split):
        X, y = self.create_test_data(time, max_cells, crop_size)

        # Get appearance and centroid tensors
        X_apps = X['appearances']
        X_cents = X['centroids']

        # Assert rotating does not mutate the shape of tensors
        rotated_X, y = tracking.random_rotate(X, y, 180)
        self.assertEqual(rotated_X['appearances'].shape, X_apps.shape)
        self.assertEqual(rotated_X['centroids'].shape, X_cents.shape)

        # Assert rotating by 0 does not change tensors
        X_apps = rotated_X['appearances'][0, 0, 0, 0]
        X_cents = rotated_X['centroids'][0, 0, 0]
        output_X, y = tracking.random_rotate(rotated_X, y, 0)
        r0 = (float)(output_X['appearances'][0, 0, 0, 0] - X_apps)
        self.assertEqual(r0, 0)
        r0 = (float)(output_X['centroids'][0, 0, 0] - X_cents)
        self.assertEqual(r0, 0)

    def test_random_translate(self, time, max_cells, crop_size,
                              batch_size, seed, track_length, val_split):
        X, y = self.create_test_data(time, max_cells, crop_size)

        # Get initial centroid values for two different cells
        init_x_0 = X['centroids'][0, 0, 0]
        init_x_1 = X['centroids'][0, 1, 0]

        translated_X, y = tracking.random_translate(X, y)

        # Get distance translated for two different cells
        r0 = (float)(translated_X['centroids'][0, 0, 0] - init_x_0)
        r1 = (float)(translated_X['centroids'][0, 1, 0] - init_x_1)
        # Assert that centroids are translated by the same amount
        assert abs(r1 - r0) < 0.001

        # Assert range of 0 does not translate data
        X_cents = translated_X['centroids'][0, 0, 0]
        output_X, y = tracking.random_translate(X, y, range=0)
        r0 = (float)(output_X['centroids'][0, 0, 0] - X_cents)
        self.assertEqual(r0, 0)

    def test_prepare_dataset(self, time, max_cells, crop_size,
                             batch_size, seed, track_length, val_split):
        # Create track_info and prepare the dataset
        track_info = self.create_track_info(226, time, max_cells, crop_size)
        train_data, val_data = tracking.prepare_dataset(track_info, rotation_range=180,
                                                        batch_size=batch_size, seed=seed,
                                                        track_length=track_length,
                                                        val_split=val_split)
        # Test data correctly batched
        for X, y in train_data.take(1):
            self.assertEqual((X['appearances'].shape)[0], batch_size)
        for X, y in val_data.take(1):
            self.assertEqual((X['appearances'].shape)[0], batch_size)

    def create_test_data(self, time, max_cells, crop_size):
        # Create dictionaries of feature data labels with correct dimensions
        X = {}
        y = {}
        appearances = np.random.random((time, max_cells, crop_size, crop_size, 1))
        X['appearances'] = tf.convert_to_tensor(appearances, dtype=tf.float32)
        centroids = np.random.random((time, max_cells, 2))
        X['centroids'] = tf.convert_to_tensor(centroids, dtype=tf.float32)
        morphologies = np.random.random((time, max_cells, 3))
        X['morphologies'] = tf.convert_to_tensor(morphologies, dtype=tf.float32)
        adj_matrices = np.random.random((time, max_cells, max_cells))
        X['adj_matrices'] = tf.convert_to_tensor(adj_matrices, dtype=tf.float32)
        temporal_adj_matrices = np.random.random((time - 1, max_cells, max_cells, 3))
        y['temporal_adj_matrices'] = tf.convert_to_tensor(temporal_adj_matrices, dtype=tf.float32)
        return X, y

    def create_track_info(self, n_batches, time, max_cells, crop_size):
        # Create track_info input (dictionary of all input and output features) with correct
        # dimensions for prepare_dataset function.
        track_info = {}
        appearances = np.random.random((n_batches, time, max_cells, crop_size, crop_size, 1))
        track_info['appearances'] = tf.convert_to_tensor(appearances, dtype=tf.float32)
        centroids = np.random.random((n_batches, time, max_cells, 2))
        track_info['centroids'] = tf.convert_to_tensor(centroids, dtype=tf.float32)
        morphologies = np.random.random((n_batches, time, max_cells, 3))
        track_info['morphologies'] = tf.convert_to_tensor(morphologies, dtype=tf.float32)
        adj_matrices = np.random.random((n_batches, time, max_cells, max_cells))
        track_info['adj_matrices'] = tf.convert_to_tensor(adj_matrices, dtype=tf.float32)
        norm_adj_matrices = np.random.random((n_batches, time, max_cells, max_cells))
        track_info['norm_adj_matrices'] = tf.convert_to_tensor(norm_adj_matrices, dtype=tf.float32)
        temporal_adj_matrices = np.random.random((n_batches, time - 1, max_cells, max_cells, 3))
        track_info['temporal_adj_matrices'] = tf.convert_to_tensor(temporal_adj_matrices,
                                                                   dtype=tf.float32)
        return track_info


if __name__ == '__main__':
    test.main()
