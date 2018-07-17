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
from deepcell.utils.train_utils import rate_scheduler


class TrainUtilsTest(test.TestCase):
    def test_rate_scheduler(self):
        # if decay is small, learning rate should decrease as epochs increase
        rs = rate_scheduler(lr=.001, decay=.95)
        assert rs(1) > rs(2)
        # if decay is large, learning rate should increase as epochs increase
        rs = rate_scheduler(lr=.001, decay=1.05)
        assert rs(1) < rs(2)
        # if decay is 1, learning rate should not change
        rs = rate_scheduler(lr=.001, decay=1)
        assert rs(1) == rs(2)

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

        img_w, img_h = 300, 300
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        img = np.random.rand(img_w, img_h, 3) * variance + bias

        # Testing that the axis_softmax function fails when passed a NumPy array.
        with self.assertRaises(AttributeError):
            axis_softmax(img, 0)
        with self.assertRaises(AttributeError):
            axis_softmax(img, 1)
        with self.assertRaises(AttributeError):
            axis_softmax(img)

if __name__ == '__main__':
    test.main()
