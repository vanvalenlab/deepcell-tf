"""Tests for tracking functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test

from deepcell.data import tracking

class TrackingTests(test.TestCase, parameterized.TestCase):
    def test_temporal_slice(self):
        time = 30
        max_cells = 25
        crop_size = 32
        track_length = 8
        X, y = self.create_test_data(time, max_cells, crop_size)
        sliced_X, sliced_y = tracking.temporal_slice(X, y, track_length)
        expected_X = {}
        expected_y = {}
        appearances = np.random.random((track_length, max_cells, crop_size, crop_size, 1))
        expected_X['appearances'] = tf.convert_to_tensor(appearances, dtype=tf.float32)
        centroids = np.random.random((track_length, max_cells, 2))
        expected_X['centroids'] = tf.convert_to_tensor(centroids, dtype=tf.float32)
        morphologies = np.random.random((track_length, max_cells, 3))
        expected_X['morphologies'] = tf.convert_to_tensor(morphologies, dtype=tf.float32)
        adj_matrices = np.random.random((track_length, max_cells, max_cells))
        expected_X['adj_matrices'] = tf.convert_to_tensor(adj_matrices, dtype=tf.float32)
        temporal_adj_matrices = np.random.random((track_length - 1, max_cells, max_cells, 3))
        expected_y['temporal_adj_matrices'] = tf.convert_to_tensor(temporal_adj_matrices, dtype=tf.float32)
        for key, data in sliced_X.items():
            self.assertEqual(sliced_X[key].shape, expected_X[key].shape)
        for key, data in sliced_y.items():
            self.assertEqual(sliced_y[key].shape, expected_y[key].shape)

    def test_random_rotate(self):
        time = 30
        max_cells = 25
        crop_size = 32
        X, y = self.create_test_data(time, max_cells, crop_size)
        rotated_X, y = tracking.random_rotate(X, y, 45)
        self.assertEqual(rotated_X['appearances'].shape, X['appearances'].shape)
        self.assertEqual(rotated_X['centroids'].shape, X['centroids'].shape)
        # theta is random, so could be zero. Unable to access theta so unable to assert 
        # whether tensors have changed.

    def test_random_translate(self):
        time = 30
        max_cells = 25
        crop_size = 32
        X, y = self.create_test_data(time, max_cells, crop_size)
        init_x_0 = X['centroids'][0,0,0]
        init_x_1 = X['centroids'][0,1,0]
        translated_X, y = tracking.random_translate(X, y)
        r0 = (float) (translated_X['centroids'][0,0,0] - init_x_0)
        r1 = (float) (translated_X['centroids'][0,1,0] - init_x_1)
        assert abs(r1 - r0) < 0.001  

    def test_prepare_dataset(self):
        time = 30
        max_cells = 25
        crop_size = 32
        batch_size = 4
        track_info, y = self.create_track_info(226, time, max_cells, crop_size)
        track_info['norm_adj_matrices'] = track_info['adj_matrices']
        track_info['temporal_adj_matrices'] = y['temporal_adj_matrices']
        for k, v in track_info.items():
            print(k, v.shape)
        train_data, val_data = tracking.prepare_dataset(track_info, rotation_range=0, batch_size=batch_size, track_length=8)
        for X, y in train_data.take(1):  
            self.assertEqual((X['appearances'].shape)[0], batch_size)
        for X, y in val_data.take(1):  
            self.assertEqual((X['appearances'].shape)[0], batch_size)

    def create_test_data(self, time, max_cells, crop_size):
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
        X = {}
        y = {}
        appearances = np.random.random((n_batches, time, max_cells, crop_size, crop_size, 1))
        X['appearances'] = tf.convert_to_tensor(appearances, dtype=tf.float32)
        centroids = np.random.random((n_batches, time, max_cells, 2))
        X['centroids'] = tf.convert_to_tensor(centroids, dtype=tf.float32)
        morphologies = np.random.random((n_batches, time, max_cells, 3))
        X['morphologies'] = tf.convert_to_tensor(morphologies, dtype=tf.float32)
        adj_matrices = np.random.random((n_batches, time, max_cells, max_cells))
        X['adj_matrices'] = tf.convert_to_tensor(adj_matrices, dtype=tf.float32)
        temporal_adj_matrices = np.random.random((n_batches, time - 1, max_cells, max_cells, 3))
        y['temporal_adj_matrices'] = tf.convert_to_tensor(temporal_adj_matrices, dtype=tf.float32)
        return X, y
        

if __name__ == '__main__':
    test.main()