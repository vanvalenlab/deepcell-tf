# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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

import pytest

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test

from deepcell_tracking.test_utils import get_annotated_movie

from deepcell.data import tracking


def get_dummy_data(num_labels=3, batches=2):
    num_labels = 3
    movies = []
    for _ in range(batches):
        movies.append(get_annotated_movie(labels_per_frame=num_labels))

    y = np.stack(movies, axis=0)
    X = np.random.random(y.shape)

    # create dummy lineage
    lineages = {}
    for b in range(X.shape[0]):
        lineages[b] = {}
        for frame in range(X.shape[1]):
            unique_labels = np.unique(y[b, frame])
            unique_labels = unique_labels[unique_labels != 0]
            for unique_label in unique_labels:
                if unique_label not in lineages[b]:
                    lineages[b][unique_label] = {
                        'frames': [frame],
                        'parent': None,
                        'daughters': [],
                        'label': unique_label,
                    }
                else:
                    lineages[b][unique_label]['frames'].append(frame)

    # tracks expect batched data
    data = {'X': X, 'y': y, 'lineages': lineages}
    return data


class TestTrack(object):

    def test_init(self, mocker):
        num_labels = 3

        data = get_dummy_data(num_labels)

        # mock reading from disk to return the expected data
        mocker.patch('deepcell.data.tracking.load_trks',
                     lambda x: data)

        track1 = tracking.Track(tracked_data=data)
        track2 = tracking.Track(path='path/to/data')

        np.testing.assert_array_equal(track1.appearances, track2.appearances)
        np.testing.assert_array_equal(
            tf.sparse.to_dense(track1.temporal_adj_matrices),
            tf.sparse.to_dense(track2.temporal_adj_matrices))

        with pytest.raises(ValueError):
            tracking.Track()

        # test if a .trk file is passed
        bad_data = {k: data[k][0] for k in data}
        bad_data['lineages'] = [bad_data['lineages']]
        with pytest.raises(ValueError):
            _ = tracking.Track(tracked_data=bad_data)

    def test_x_y_padding(self):
        num_labels = 3

        data = get_dummy_data(num_labels)

        pads = [(0, 5) if i in {2, 3} else (0, 0) for i in range(data['y'].ndim)]

        padded_data = {
            'X': np.pad(data['X'], pads, mode='constant'),
            'y': np.pad(data['y'], pads, mode='constant'),
            'lineages': data['lineages'],
        }

        unpadded = tracking.Track(tracked_data=data)
        padded = tracking.Track(tracked_data=padded_data)

        np.testing.assert_array_equal(unpadded.appearances, padded.appearances)
        np.testing.assert_array_equal(
            tf.sparse.to_dense(unpadded.temporal_adj_matrices),
            tf.sparse.to_dense(padded.temporal_adj_matrices))

    def test_concat_tracks(self):
        num_labels = 3

        data = get_dummy_data(num_labels)
        track_1 = tracking.Track(tracked_data=data)
        track_2 = tracking.Track(tracked_data=data)

        data = tracking.concat_tracks([track_1, track_2])

        for k, v in data.items():
            starting_batch = 0
            for t in (track_1, track_2):
                assert hasattr(t, k)
                w = getattr(t, k)
                if not isinstance(w, tf.sparse.SparseTensor):
                    # data is put into top left corner of array
                    v_sub = v[
                        starting_batch:starting_batch + w.shape[0],
                        0:w.shape[1],
                        0:w.shape[2],
                        0:w.shape[3]
                    ]
                    np.testing.assert_array_equal(v_sub, w)
                # TODO: how to test the sparse tensors?

        # test that input must be iterable
        with pytest.raises(TypeError):
            tracking.concat_tracks(track_1)


@parameterized.named_parameters([
    {
        'testcase_name': 'test_data_01',
        'time': 10,
        'max_cells': 8,
        'crop_size': 32,
        'batch_size': 2,
        'seed': None,
        'track_length': 4,
        'val_size': 0.15,
        'test_size': 0.15
    }, {
        'testcase_name': 'test_data_02',
        'time': 13,
        'max_cells': 15,
        'crop_size': 32,
        'batch_size': 2,
        'seed': 2,
        'track_length': 4,
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
        n_batches = 10
        track_info = {}
        track_info['appearances'] = tf.random.uniform(
            [n_batches, time, max_cells, crop_size, crop_size, 1], 0, 1)
        track_info['centroids'] = tf.random.uniform(
            [n_batches, time, max_cells, 2], 0, 512)
        track_info['morphologies'] = tf.random.uniform(
            [n_batches, time, max_cells, 3], 0, 1)
        track_info['adj_matrices'] = tf.sparse.from_dense(tf.random.uniform(
            [n_batches, time, max_cells, max_cells], 0, 1))
        track_info['norm_adj_matrices'] = tf.sparse.from_dense(tf.random.uniform(
            [n_batches, time, max_cells, max_cells], 0, 1))
        track_info['temporal_adj_matrices'] = tf.sparse.from_dense(tf.random.uniform(
            [n_batches, time - 1, max_cells, max_cells, 3], 0, 1))

        train_data, val_data, test_data = tracking.prepare_dataset(
            track_info,
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
        X['morphologies'] = tf.sparse.from_dense(
            tf.random.uniform([time, max_cells, 3], 0, 1))
        X['adj_matrices'] = tf.sparse.from_dense(
            tf.random.uniform([time, max_cells, max_cells], 0, 1))
        y['temporal_adj_matrices'] = tf.sparse.from_dense(
            tf.random.uniform([time - 1, max_cells, max_cells, 3], 0, 1))
        return X, y


if __name__ == '__main__':
    test.main()
