"""
Tests for data_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.platform import test

from deepcell.utils.data_utils import relabel_movie
from deepcell.utils.data_utils import reshape_movie
from deepcell.utils.data_utils import reshape_matrix


class TestDataUtils(test.TestCase):

    def test_relabel_movie(self):
        y = np.array([[0, 3, 5], [4, 99, 123]])
        assert np.array_equal(relabel_movie(y), np.array([[0, 1, 3], [2, 4, 5]]))

    def test_reshape_movie(self):
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

    def test_reshape_matrix(self):
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
            new_X, new_y = reshape_movie(smaller, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(bigger, y, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(X, smaller, new_size)
        with self.assertRaises(ValueError):
            new_X, new_y = reshape_movie(X, bigger, new_size)


if __name__ == '__main__':
    test.main()
