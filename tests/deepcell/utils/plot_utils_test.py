"""
Tests for io_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.platform import test

from deepcell.utils.plot_utils import cf


class PlotUtilsTest(test.TestCase):
    def test_cf(self):
        img_w, img_h = 300, 300
        bias = np.random.rand(img_w, img_h) * 64
        variance = np.random.rand(img_w, img_h) * (255 - 64)
        img = np.random.rand(img_w, img_h) * variance + bias

        # values are hard-coded for test image
        shape = img.shape
        # test coordinates outside of test_img dimensions
        assert cf(shape[0] + 1, shape[1] + 1, img) == 'x=301.0000, y=301.0000'
        assert cf(-1 * shape[0], -1 * shape[1], img) == 'x=-300.0000, y=-300.0000'
        # test coordinates inside test_img dimensions
        assert cf(shape[0] / 2, shape[1] / 2, img) == 'x=150.0000, y=150.0000, z=93.0260'

if __name__ == '__main__':
    test.main()
