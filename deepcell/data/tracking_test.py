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
        'val_size': 0.15,
        'test_size': 0.15
    }, {
        'testcase_name': 'test_data_02',
        'time': 40,
        'max_cells': 29,
        'crop_size': 32,
        'batch_size': 24,
        'seed': 2,
        'track_length': 7,
        'val_size': 0.2,
        'test_size': 0
    }
])
class TrackingTests(test.TestCase, parameterized.TestCase):

    def test_temporal_slice(self, time, max_cells, crop_size,
                            batch_size, seed, track_length,
                            val_size, test_size):
        X, y = self.create_test_data(time, max_cells, crop_size)
        sliced_X, sliced_y = tracking.temporal_slice(X, y, track_length)

        # Test temporal dimension correct index and value
        for key in sliced_X:
            self.assertEqual(sliced_X[key].shape[0], track_length)
        for key in sliced_y:
            self.assertEqual(sliced_y[key].shape[0], track_length - 1)

    def test_random_rotate(self, time, max_cells, crop_size,
                           batch_size, seed, track_length,
                           val_size, test_size):
        X, y = self.create_test_data(time, max_cells, crop_size)

        # Get appearance and centroid tensors
        X_apps = X['appearances']
        X_cents = X['centroids']

        # Assert rotating does not mutate the shape of tensors
        rotated_X, y = tracking.random_rotate(X, y, 180)
        self.assertEqual(rotated_X['appearances'].shape, X_apps.shape)
        self.assertEqual(rotated_X['centroids'].shape, X_cents.shape)

        # Assert rotating by 0 does not change tensors
        X_apps = rotated_X['appearances']
        X_cents = rotated_X['centroids']

        output_X, y = tracking.random_rotate(rotated_X, y, 0)

        self.assertAllEqual(output_X['appearances'], X_apps)
        self.assertAllEqual(output_X['centroids'], X_cents)

    def test_random_translate(self, time, max_cells, crop_size,
                              batch_size, seed, track_length,
                              val_size, test_size):
        X, y = self.create_test_data(time, max_cells, crop_size)
        X_cents = X['centroids']
        translated_X, y = tracking.random_translate(X, y)

        # Get difference from translating
        diff = X_cents - translated_X['centroids']

        # Create array filled with same x and y values
        x0 = np.empty((time, max_cells, 1))
        y0 = np.empty((time, max_cells, 1))
        x0.fill(diff[0, 0, 0])
        y0.fill(diff[0, 0, 1])
        r0 = tf.concat([x0, y0], 2)

        # Assert that centroids are translated by the same amount
        self.assertAllEqual(diff, r0)

        # Assert range of 0 does not translate data
        X_cents = translated_X['centroids']
        output_X, y = tracking.random_translate(X, y, range=0)
        self.assertAllEqual(output_X['centroids'], X_cents)

    def test_prepare_dataset(self, time, max_cells, crop_size,
                             batch_size, seed, track_length,
                             val_size, test_size):
        # Create track_info and prepare the dataset
        track_info = self.create_track_info(226, time, max_cells, crop_size)
        train_data, val_data, test_data = tracking.prepare_dataset(track_info,
                                                                   rotation_range=180,
                                                                   batch_size=batch_size,
                                                                   seed=seed,
                                                                   track_length=track_length,
                                                                   val_size=val_size,
                                                                   test_size=test_size)
        # Test data correctly batched
        for X, y in train_data.take(1):
            self.assertEqual((X['appearances'].shape)[0], batch_size)
        for X, y in val_data.take(1):
            self.assertEqual((X['appearances'].shape)[0], batch_size)
        if test_size != 0:
            for X, y in test_data.take(1):
                self.assertEqual((X['appearances'].shape)[0], batch_size)

    def create_test_data(self, time, max_cells, crop_size):
        # Create dictionaries of feature data labels with correct dimensions
        X = {}
        y = {}
        X['appearances'] = tf.random.uniform([time, max_cells, crop_size, crop_size, 1], 0, 1)
        X['centroids'] = tf.random.uniform([time, max_cells, 2], 0, 512)
        X['morphologies'] = tf.random.uniform([time, max_cells, 3], 0, 1)
        X['adj_matrices'] = tf.random.uniform([time, max_cells, max_cells], 0, 1)
        y['temporal_adj_matrices'] = tf.random.uniform([time - 1, max_cells, max_cells, 3], 0, 1)
        return X, y

    def create_track_info(self, n_batches, time, max_cells, crop_size):
        # Create track_info input (dictionary of all input and output features) with correct
        # dimensions for prepare_dataset function.
        track_info = {}
        track_info['appearances'] = tf.random.uniform([n_batches, time, max_cells,
                                                       crop_size, crop_size, 1], 0, 1)
        track_info['centroids'] = tf.random.uniform([n_batches, time, max_cells, 2], 0, 512)
        track_info['morphologies'] = tf.random.uniform([n_batches, time, max_cells, 3], 0, 1)
        track_info['adj_matrices'] = tf.random.uniform([n_batches, time, max_cells, max_cells],
                                                       0, 1)
        track_info['norm_adj_matrices'] = tf.random.uniform([n_batches, time, max_cells,
                                                            max_cells], 0, 1)
        track_info['temporal_adj_matrices'] = tf.random.uniform([n_batches, time - 1, max_cells,
                                                                max_cells, 3], 0, 1)
        return track_info


if __name__ == '__main__':
    test.main()
