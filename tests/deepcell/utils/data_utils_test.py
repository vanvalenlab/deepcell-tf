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
"""Tests for data_utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from deepcell.utils import data_utils


class TestDataUtils(test.TestCase):

    def test_get_data(self):
        test_size = .1
        img_w, img_h = 30, 30
        X = np.random.random((10, img_w, img_h, 1))
        y = np.random.randint(3, size=(10, img_w, img_h, 1))

        temp_dir = self.get_temp_dir()
        good_file = os.path.join(temp_dir, 'good.npz')
        np.savez(good_file, X=X, y=y)

        train_dict, test_dict = data_utils.get_data(
            good_file, test_size=test_size)

        X_test, X_train = test_dict['X'], train_dict['X']

        self.assertIsInstance(train_dict, dict)
        self.assertIsInstance(test_dict, dict)

        self.assertEqual(X_test.size / (X_test.size + X_train.size), test_size)

        bad_file = os.path.join(temp_dir, 'bad.npz')
        np.savez(bad_file, X_bad=X, y_bad=y)
        with self.assertRaises(KeyError):
            _, _ = data_utils.get_data(bad_file)

    def test_get_max_sample_num_list(self):
        K.set_image_data_format('channels_last')
        edge_feature = [1, 0, 0]  # first channel index is cell edge
        win_x, win_y = 10, 10
        y = np.zeros((2, 30, 30, 3))
        y[:, 0, 0, 0] = 1  # set value outside valid window range
        y[:, win_x + 1, win_y + 1, 0] = 1  # set value inside valid window range
        # test non-sample mode
        max_nums = data_utils.get_max_sample_num_list(y, edge_feature,
                                                      output_mode='conv',
                                                      padding='same',
                                                      window_size_x=win_x,
                                                      window_size_y=win_y)
        self.assertEqual(max_nums, [np.Inf, np.Inf])

        # test sample mode, no padding
        max_nums = data_utils.get_max_sample_num_list(y, edge_feature,
                                                      output_mode='sample',
                                                      padding='same',
                                                      window_size_x=win_x,
                                                      window_size_y=win_y)
        self.assertEqual(max_nums, [2, 2])

        # test sample mode, valid padding
        max_nums = data_utils.get_max_sample_num_list(y, edge_feature,
                                                      output_mode='sample',
                                                      padding='valid',
                                                      window_size_x=win_x,
                                                      window_size_y=win_y)
        self.assertEqual(max_nums, [1, 1])

        # channels_first
        K.set_image_data_format('channels_first')
        edge_feature = [1, 0, 0]  # first channel index is cell edge
        win_x, win_y = 10, 10
        y = np.zeros((2, 3, 30, 30))
        y[:, 0, 0, 0] = 1  # set value outside valid window range
        y[:, 0, win_x + 1, win_y + 1] = 1  # set value inside valid window range

        # test sample mode, no padding
        max_nums = data_utils.get_max_sample_num_list(y, edge_feature,
                                                      output_mode='sample',
                                                      padding='same',
                                                      window_size_x=win_x,
                                                      window_size_y=win_y)
        self.assertEqual(max_nums, [2, 2])

        # test sample mode, valid padding
        max_nums = data_utils.get_max_sample_num_list(y, edge_feature,
                                                      output_mode='sample',
                                                      padding='valid',
                                                      window_size_x=win_x,
                                                      window_size_y=win_y)
        self.assertEqual(max_nums, [1, 1])

    def test_sample_label_matrix(self):
        win_x, win_y = 10, 10
        y = np.zeros((2, 30, 30, 3))
        y[:, 0, 0, 0] = 1  # set value outside valid window range
        y[:, win_x + 1, win_y + 1, 0] = 1  # set value inside valid window range
        r, c, b, l = data_utils.sample_label_matrix(
            y, window_size=(win_x, win_y),
            padding='valid', data_format='channels_last')
        self.assertListEqual(list(map(len, [r, c, b, l])), [len(r)] * 4)
        self.assertEqual(np.unique(b).size, 2)
        self.assertEqual([np.unique(r).size, np.unique(c).size], [1, 1])
        self.assertEqual(np.unique(l).size, 1)

        r, c, b, l = data_utils.sample_label_matrix(
            y, window_size=(win_x, win_y),
            max_training_examples=None,
            padding='same', data_format='channels_last')
        self.assertListEqual(list(map(len, [r, c, b, l])), [len(r)] * 4)
        self.assertEqual([np.unique(r).size, np.unique(c).size], [2, 2])
        self.assertEqual(np.unique(l).size, 1)

        # test channels_first
        win_x, win_y = 10, 10
        y = np.zeros((2, 3, 30, 30))
        y[:, 0, 0, 0] = 1  # set value outside valid window range
        y[:, 0, win_x + 1, win_y + 1] = 1  # set value inside valid window range
        r, c, b, l = data_utils.sample_label_matrix(
            y, window_size=(win_x, win_y),
            padding='valid', data_format='channels_first')
        self.assertListEqual(list(map(len, [r, c, b, l])), [len(r)] * 4)
        self.assertEqual(np.unique(b).size, 2)
        self.assertEqual([np.unique(r).size, np.unique(c).size], [1, 1])
        self.assertEqual(np.unique(l).size, 1)

        r, c, b, l = data_utils.sample_label_matrix(
            y, window_size=(win_x, win_y),
            max_training_examples=None,
            padding='same', data_format='channels_first')
        self.assertListEqual(list(map(len, [r, c, b, l])), [len(r)] * 4)
        self.assertEqual(np.unique(b).size, 2)
        self.assertEqual([np.unique(r).size, np.unique(c).size], [2, 2])
        self.assertEqual(np.unique(l).size, 1)

    def test_sample_label_movie(self):
        win_x, win_y, win_z = 10, 10, 1
        y = np.zeros((2, 5, 30, 30, 3))
        # set value outside valid window range
        y[:, 0, 0, 0, 0] = 1
        # set value inside valid window range
        y[:, win_z + 1, win_x + 1, win_y + 1, 0] = 1
        f, r, c, b, l = data_utils.sample_label_movie(
            y, window_size=(win_x, win_y, win_z),
            padding='valid', data_format='channels_last')
        self.assertListEqual(list(map(len, [r, c, b, l, f])), [len(r)] * 5)
        self.assertEqual(np.unique(b).size, 2)
        self.assertEqual([np.unique(r).size,
                          np.unique(c).size,
                          np.unique(f).size],
                         [1, 1, 1])
        self.assertEqual(np.unique(l).size, 1)

        f, r, c, b, l = data_utils.sample_label_movie(
            y, window_size=(win_x, win_y, win_z),
            max_training_examples=None,
            padding='same', data_format='channels_last')
        self.assertListEqual(list(map(len, [r, c, b, l, f])), [len(r)] * 5)
        self.assertEqual([np.unique(r).size,
                          np.unique(c).size,
                          np.unique(f).size],
                         [2, 2, 2])
        self.assertEqual(np.unique(l).size, 1)

        # test channels_first
        win_x, win_y, win_z = 10, 10, 1
        y = np.zeros((2, 3, 5, 30, 30))
        # set value outside valid window range
        y[:, 0, 0, 0, 0] = 1
        # set value inside valid window range
        y[:, 0, win_z + 1, win_x + 1, win_y + 1] = 1
        f, r, c, b, l = data_utils.sample_label_movie(
            y, window_size=(win_x, win_y, win_z),
            padding='valid', data_format='channels_first')
        self.assertListEqual(list(map(len, [r, c, b, l, f])), [len(r)] * 5)
        self.assertEqual(np.unique(b).size, 2)
        self.assertEqual([np.unique(r).size, np.unique(c).size], [1, 1])
        self.assertEqual(np.unique(l).size, 1)

        f, r, c, b, l = data_utils.sample_label_movie(
            y, window_size=(win_x, win_y, win_z),
            max_training_examples=None,
            padding='same', data_format='channels_first')
        self.assertListEqual(list(map(len, [r, c, b, l, f])), [len(r)] * 5)
        self.assertEqual(np.unique(b).size, 2)
        self.assertEqual([np.unique(r).size, np.unique(c).size], [2, 2])
        self.assertEqual(np.unique(l).size, 1)

    def test_trim_padding(self):
        # test 2d image
        K.set_image_data_format('channels_last')
        img_size = 512
        win_x, win_y = 30, 30
        x_trim = img_size - 2 * win_x
        y_trim = img_size - 2 * win_y
        K.set_image_data_format('channels_last')
        arr = np.zeros((1, img_size, img_size, 1))
        arr_trim = data_utils.trim_padding(arr, win_x, win_y)
        self.assertEqual(arr_trim.shape, (1, x_trim, y_trim, 1))
        # test channels_first
        K.set_image_data_format('channels_first')
        arr = np.zeros((1, 1, img_size, img_size))
        arr_trim = data_utils.trim_padding(arr, win_x, win_y)
        self.assertEqual(arr_trim.shape, (1, 1, x_trim, y_trim))

        # test 3d image stack
        img_size = 256
        frames = 30
        win_x, win_y = 20, 30
        win_z = 2
        x_trim = img_size - 2 * win_x
        y_trim = img_size - 2 * win_y
        z_trim = frames - 2 * win_z
        K.set_image_data_format('channels_last')
        arr = np.zeros((1, frames, img_size, img_size, 1))
        # trim win_z
        arr_trim = data_utils.trim_padding(arr, win_x, win_y, win_z)
        self.assertEqual(arr_trim.shape, (1, z_trim, x_trim, y_trim, 1))
        # don't trim win_z
        arr_trim = data_utils.trim_padding(arr, win_x, win_y)
        self.assertEqual(arr_trim.shape, (1, frames, x_trim, y_trim, 1))
        # test channels_first
        K.set_image_data_format('channels_first')
        arr = np.zeros((1, 1, 30, img_size, img_size))
        # trim win_z
        arr_trim = data_utils.trim_padding(arr, win_x, win_y, win_z)
        self.assertEqual(arr_trim.shape, (1, 1, z_trim, x_trim, y_trim))
        # don't trim win_z
        arr_trim = data_utils.trim_padding(arr, win_x, win_y)
        self.assertEqual(arr_trim.shape, (1, 1, frames, x_trim, y_trim))

        # test bad input
        with self.assertRaises(ValueError):
            small_arr = np.zeros((img_size, img_size, 1))
            data_utils.trim_padding(small_arr, 10, 10)
        with self.assertRaises(ValueError):
            big_arr = np.zeros((1, 1, 30, img_size, img_size, 1))
            data_utils.trim_padding(big_arr, 10, 10)

    def test_relabel_movie(self):
        y = np.array([[0, 3, 5], [4, 99, 123]])
        relabeled = data_utils.relabel_movie(y)
        self.assertAllEqual(relabeled, np.array([[0, 1, 3], [2, 4, 5]]))

    def test_reshape_movie(self):
        K.set_image_data_format('channels_last')
        X = np.zeros((1, 3, 16, 16, 3))
        y = np.zeros((1, 3, 16, 16, 1))
        new_size = 4

        # test resize to smaller image, divisible
        new_X, new_y = data_utils.reshape_movie(X, y, new_size)
        new_batch = np.ceil(16 / new_size) ** 2
        self.assertEqual(new_X.shape, (new_batch, 3, new_size, new_size, 3))
        self.assertEqual(new_y.shape, (new_batch, 3, new_size, new_size, 1))

        # test reshape with non-divisible values.
        new_size = 5
        new_batch = np.ceil(16 / new_size) ** 2
        new_X, new_y = data_utils.reshape_movie(X, y, new_size)
        self.assertEqual(new_X.shape, (new_batch, 3, new_size, new_size, 3))
        self.assertEqual(new_y.shape, (new_batch, 3, new_size, new_size, 1))

        # test reshape to bigger size
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_movie(X, y, 32)

        # test wrong dimensions
        bigger = np.zeros((1, 3, 16, 16, 3, 1))
        smaller = np.zeros((1, 16, 16, 3))
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_movie(smaller, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_movie(bigger, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_movie(X, smaller, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_movie(X, bigger, new_size)

        # channels_first
        K.set_image_data_format('channels_first')
        X = np.zeros((1, 3, 3, 16, 16))
        y = np.zeros((1, 1, 3, 16, 16))
        new_size = 4

        # test resize to smaller image, divisible
        new_X, new_y = data_utils.reshape_movie(X, y, new_size)
        new_batch = np.ceil(16 / new_size) ** 2
        self.assertEqual(new_X.shape, (new_batch, 3, 3, new_size, new_size))
        self.assertEqual(new_y.shape, (new_batch, 1, 3, new_size, new_size))

        # test reshape with non-divisible values.
        new_size = 5
        new_batch = np.ceil(16 / new_size) ** 2
        new_X, new_y = data_utils.reshape_movie(X, y, new_size)
        self.assertEqual(new_X.shape, (new_batch, 3, 3, new_size, new_size))
        self.assertEqual(new_y.shape, (new_batch, 1, 3, new_size, new_size))

    def test_reshape_matrix(self):
        K.set_image_data_format('channels_last')
        X = np.zeros((1, 16, 16, 3))
        y = np.zeros((1, 16, 16, 1))
        new_size = 4

        # test resize to smaller image, divisible
        new_X, new_y = data_utils.reshape_matrix(X, y, new_size)
        new_batch = np.ceil(16 / new_size) ** 2
        self.assertEqual(new_X.shape, (new_batch, new_size, new_size, 3))
        self.assertEqual(new_y.shape, (new_batch, new_size, new_size, 1))

        # test reshape with non-divisible values.
        new_size = 5
        new_batch = np.ceil(16 / new_size) ** 2
        new_X, new_y = data_utils.reshape_matrix(X, y, new_size)
        self.assertEqual(new_X.shape, (new_batch, new_size, new_size, 3))
        self.assertEqual(new_y.shape, (new_batch, new_size, new_size, 1))

        # test reshape to bigger size
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_matrix(X, y, 32)

        # test wrong dimensions
        bigger = np.zeros((1, 16, 16, 3, 1))
        smaller = np.zeros((1, 16, 16))
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_matrix(smaller, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_matrix(bigger, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_matrix(X, smaller, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = data_utils.reshape_matrix(X, bigger, new_size)

        # channels_first
        K.set_image_data_format('channels_first')
        X = np.zeros((1, 3, 16, 16))
        y = np.zeros((1, 1, 16, 16))
        new_size = 4

        # test resize to smaller image, divisible
        new_X, new_y = data_utils.reshape_matrix(X, y, new_size)
        new_batch = np.ceil(16 / new_size) ** 2
        self.assertEqual(new_X.shape, (new_batch, 3, new_size, new_size))
        self.assertEqual(new_y.shape, (new_batch, 1, new_size, new_size))

        # test reshape with non-divisible values.
        new_size = 5
        new_batch = np.ceil(16 / new_size) ** 2
        new_X, new_y = data_utils.reshape_matrix(X, y, new_size)
        self.assertEqual(new_X.shape, (new_batch, 3, new_size, new_size))
        self.assertEqual(new_y.shape, (new_batch, 1, new_size, new_size))


if __name__ == '__main__':
    test.main()
