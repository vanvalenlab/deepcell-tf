"""
Tests for data_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from deepcell.utils.data_utils import trim_padding
from deepcell.utils.data_utils import relabel_movie
from deepcell.utils.data_utils import reshape_movie
from deepcell.utils.data_utils import reshape_matrix


class TestDataUtils(test.TestCase):

    def test_trim_padding(self):
        # test 2d image
        K.set_image_data_format('channels_last')
        img_size = 512
        win_x, win_y = 30, 30
        trimmed_x = img_size - 2 * win_x
        trimmed_y = img_size - 2 * win_y
        K.set_image_data_format('channels_last')
        arr = np.zeros((1, img_size, img_size, 1))
        trimmed_arr = trim_padding(arr, win_x, win_y)
        assert trimmed_arr.shape == (1, trimmed_x, trimmed_y, 1)
        # test channels_first
        K.set_image_data_format('channels_first')
        arr = np.zeros((1, 1, img_size, img_size))
        trimmed_arr = trim_padding(arr, win_x, win_y)
        assert trimmed_arr.shape == (1, 1, trimmed_x, trimmed_y)

        # test 3d image stack
        img_size = 256
        win_x, win_y = 20, 30
        trimmed_x = img_size - 2 * win_x
        trimmed_y = img_size - 2 * win_y
        K.set_image_data_format('channels_last')
        arr = np.zeros((1, 30, img_size, img_size, 1))
        trimmed_arr = trim_padding(arr, win_x, win_y)
        assert trimmed_arr.shape == (1, 30, trimmed_x, trimmed_y, 1)
        # test channels_first
        K.set_image_data_format('channels_first')
        arr = np.zeros((1, 1, 30, img_size, img_size))
        trimmed_arr = trim_padding(arr, win_x, win_y)
        assert trimmed_arr.shape == (1, 1, 30, trimmed_x, trimmed_y)

        # test bad input
        with self.assertRaises(ValueError):
            small_arr = np.zeros((img_size, img_size, 1))
            trim_padding(small_arr, 10, 10)
        with self.assertRaises(ValueError):
            big_arr = np.zeros((1, 1, 30, img_size, img_size, 1))
            trim_padding(big_arr, 10, 10)

    def test_relabel_movie(self):
        y = np.array([[0, 3, 5], [4, 99, 123]])
        assert np.array_equal(relabel_movie(y), np.array([[0, 1, 3], [2, 4, 5]]))

    def test_reshape_movie(self):
        K.set_image_data_format('channels_last')
        X = np.zeros((1, 30, 1024, 1024, 3))
        y = np.zeros((1, 30, 1024, 1024, 1))
        new_size = 256

        # test resize to smaller image, divisible
        new_X, new_y = reshape_movie(X, y, new_size)
        new_batch = np.ceil(1024 / new_size) ** 2
        assert new_X.shape == (new_batch, 30, new_size, new_size, 3)
        assert new_y.shape == (new_batch, 30, new_size, new_size, 1)

        # test reshape with non-divisible values.
        new_size = 200
        new_batch = np.ceil(1024 / new_size) ** 2
        new_X, new_y = reshape_movie(X, y, new_size)
        assert new_X.shape == (new_batch, 30, new_size, new_size, 3)
        assert new_y.shape == (new_batch, 30, new_size, new_size, 1)

        # test reshape to bigger size
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(X, y, 2048)

        # test wrong dimensions
        bigger = np.zeros((1, 30, 1024, 1024, 3, 1))
        smaller = np.zeros((1, 1024, 1024, 3))
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(smaller, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(bigger, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(X, smaller, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(X, bigger, new_size)

    def test_reshape_movie_channels_first(self):
        K.set_image_data_format('channels_first')
        X = np.zeros((1, 3, 30, 1024, 1024))
        y = np.zeros((1, 1, 30, 1024, 1024))
        new_size = 256

        # test resize to smaller image, divisible
        new_X, new_y = reshape_movie(X, y, new_size)
        new_batch = np.ceil(1024 / new_size) ** 2
        assert new_X.shape == (new_batch, 3, 30, new_size, new_size)
        assert new_y.shape == (new_batch, 1, 30, new_size, new_size)

        # test reshape with non-divisible values.
        new_size = 200
        new_batch = np.ceil(1024 / new_size) ** 2
        new_X, new_y = reshape_movie(X, y, new_size)
        assert new_X.shape == (new_batch, 3, 30, new_size, new_size)
        assert new_y.shape == (new_batch, 1, 30, new_size, new_size)

    def test_reshape_matrix(self):
        K.set_image_data_format('channels_last')
        X = np.zeros((1, 1024, 1024, 3))
        y = np.zeros((1, 1024, 1024, 1))
        new_size = 256

        # test resize to smaller image, divisible
        new_X, new_y = reshape_matrix(X, y, new_size)
        new_batch = np.ceil(1024 / new_size) ** 2
        assert new_X.shape == (new_batch, new_size, new_size, 3)
        assert new_y.shape == (new_batch, new_size, new_size, 1)

        # test reshape with non-divisible values.
        new_size = 200
        new_batch = np.ceil(1024 / new_size) ** 2
        new_X, new_y = reshape_matrix(X, y, new_size)
        assert new_X.shape == (new_batch, new_size, new_size, 3)
        assert new_y.shape == (new_batch, new_size, new_size, 1)

        # test reshape to bigger size
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_matrix(X, y, 2048)

        # test wrong dimensions
        bigger = np.zeros((1, 1024, 1024, 3, 1))
        smaller = np.zeros((1, 1024, 1024))
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_matrix(smaller, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_matrix(bigger, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_matrix(X, smaller, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_matrix(X, bigger, new_size)

    def test_reshape_matrix_channels_first(self):
        K.set_image_data_format('channels_first')
        X = np.zeros((1, 3, 1024, 1024))
        y = np.zeros((1, 1, 1024, 1024))
        new_size = 256

        # test resize to smaller image, divisible
        new_X, new_y = reshape_matrix(X, y, new_size)
        new_batch = np.ceil(1024 / new_size) ** 2
        assert new_X.shape == (new_batch, 3, new_size, new_size)
        assert new_y.shape == (new_batch, 1, new_size, new_size)

        # test reshape with non-divisible values.
        new_size = 200
        new_batch = np.ceil(1024 / new_size) ** 2
        new_X, new_y = reshape_matrix(X, y, new_size)
        assert new_X.shape == (new_batch, 3, new_size, new_size)
        assert new_y.shape == (new_batch, 1, new_size, new_size)


if __name__ == '__main__':
    test.main()
