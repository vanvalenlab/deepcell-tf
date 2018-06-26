"""
Tests for train_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from skimage.io import imread
from tensorflow.python import keras
from tensorflow.python.platform import test

from deepcell.utils.train_utils import axis_softmax

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(TEST_DIR, 'resources')
TEST_IMG = imread(os.path.join(RES_DIR, 'phase.tif'))

class TrainUtilsTest(test.TestCase):
    def test_axis_softmax(self):
        """
        Adapted from the Tensorflow test for the softmax layer.
        """

        def _ref_softmax(values):
            m = np.max(values)
            e = np.exp(values - m)
            return e / np.sum(e)

        with self.test_session():
            x = keras.backend.placeholder(ndim=2)
            f = keras.backend.function([x], [keras.activations.softmax(x)])
            test_values = np.random.random((2, 5))

            result = f([test_values])[0]
        expected = _ref_softmax(test_values[0])
        self.assertAllClose(result[0], expected, rtol=1e-05)

        with self.assertRaises(ValueError):
            x = keras.backend.placeholder(ndim=1)
            keras.activations.softmax(x)

        # Testing that the axis_softmax function fails when passed a NumPy array.
        with self.assertRaises(AttributeError):
            axis_softmax(TEST_IMG, 0)
        with self.assertRaises(AttributeError):
            axis_softmax(TEST_IMG, 1)
        with self.assertRaises(AttributeError):
            axis_softmax(TEST_IMG)

if __name__ == '__main__':
    test.main()
