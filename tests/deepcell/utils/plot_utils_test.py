# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""Tests for io_utils"""
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
        self.assertEqual(cf(shape[0] + 1, shape[1] + 1, img), 'x=301.0000, y=301.0000')
        self.assertEqual(cf(-1 * shape[0], -1 * shape[1], img), 'x=-300.0000, y=-300.0000')
        # test coordinates inside test_img dimensions
        self.assertEqual(cf(shape[0] / 2, shape[1] / 2, img), 'x=150.0000, y=150.0000, z=93.0260')

if __name__ == '__main__':
    test.main()
