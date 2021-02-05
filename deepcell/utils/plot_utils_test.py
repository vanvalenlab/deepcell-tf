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
"""Tests for io_utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest

import numpy as np
from tensorflow.python.platform import test
from skimage.segmentation import find_boundaries

from deepcell.utils import plot_utils


class PlotUtilsTest(test.TestCase):
    def test_cf(self):
        img_w, img_h = 300, 300
        bias = np.random.rand(img_w, img_h) * 64
        variance = np.random.rand(img_w, img_h) * (255 - 64)
        img = np.random.rand(img_w, img_h) * variance + bias

        # values are hard-coded for test image
        shape = img.shape
        # test coordinates outside of test_img dimensions
        label = plot_utils.cf(shape[0] + 1, shape[1] + 1, img)
        self.assertEqual(label, 'x=301.0000, y=301.0000')
        label = plot_utils.cf(-1 * shape[0], -1 * shape[1], img)
        self.assertEqual(label, 'x=-300.0000, y=-300.0000')
        # test coordinates inside test_img dimensions
        label = plot_utils.cf(shape[0] / 2, shape[1] / 2, img)
        self.assertEqual(label, 'x=150.0000, y=150.0000, z=93.0260')

if __name__ == '__main__':
    test.main()


def test_create_rgb_image():
    test_input = np.random.rand(2, 50, 50, 2)

    rgb_output = plot_utils.create_rgb_image(input_data=test_input,
                                             channel_colors=['red', 'green'])

    # blue channel is empty
    assert np.sum(rgb_output[..., 0]) > 0
    assert np.sum(rgb_output[..., 1]) > 0
    assert np.sum(rgb_output[..., 2]) == 0

    rgb_output = plot_utils.create_rgb_image(input_data=test_input[..., :1],
                                             channel_colors=['blue'])

    assert np.sum(rgb_output[..., 0]) == 0
    assert np.sum(rgb_output[..., 1]) == 0
    assert np.sum(rgb_output[..., 2]) > 0

    # invalid input shape
    with pytest.raises(ValueError):
        _ = plot_utils.create_rgb_image(input_data=test_input[0], channel_colors=['red', 'green'])

    # too many channels
    double_input = np.concatenate((test_input, test_input), axis=-1)
    with pytest.raises(ValueError):
        _ = plot_utils.create_rgb_image(input_data=double_input,
                                        channel_colors=['red', 'green'])

    # invalid channel name
    with pytest.raises(ValueError):
        _ = plot_utils.create_rgb_image(input_data=test_input,
                                        channel_colors=['red', 'purple'])

    # not enough channel names
    with pytest.raises(ValueError):
        _ = plot_utils.create_rgb_image(input_data=test_input,
                                        channel_colors=['red'])


def test_make_outline_overlay():
    rgb_data = np.random.rand(2, 50, 50, 3)

    predictions = np.zeros((2, 50, 50, 1), dtype='int')
    predictions[0, :10, :10, 0] = 1
    predictions[0, 15:30, 30:45, 0] = 2
    predictions[1, 10:15, 25:35, 0] = 1
    predictions[1, 40:50, 0:10, 0] = 2

    overlay = plot_utils.make_outline_overlay(rgb_data=rgb_data, predictions=predictions)
    for img in range(predictions.shape[0]):
        outline = find_boundaries(predictions[img, ..., 0], connectivity=1, mode='inner')
        outline_mask = outline > 0
        assert np.all(overlay[img, outline_mask, 0] == 1)

    # invalid prediction shape
    with pytest.raises(ValueError):
        _ = plot_utils.make_outline_overlay(rgb_data=rgb_data, predictions=predictions[0])

    # more predictions than rgb images
    with pytest.raises(ValueError):
        _ = plot_utils.make_outline_overlay(rgb_data=rgb_data[:1], predictions=predictions)
