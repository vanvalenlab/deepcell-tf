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
"""Tests for retinanet_anchor_utils"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import configparser

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.platform import test

from deepcell.utils.retinanet_anchor_utils import anchors_for_shape
from deepcell.utils.retinanet_anchor_utils import AnchorParameters
# from deepcell.utils.retinanet_anchor_utils import read_config_file
# from deepcell.utils.retinanet_anchor_utils import parse_anchor_parameters


class TestRetinaNetAnchorUtils(test.TestCase):

    # def test_config_read(self):
    #     config = read_config_file('tests/test-data/config/config.ini')
    #     assert 'anchor_parameters' in config
    #     assert 'sizes' in config['anchor_parameters']
    #     assert 'strides' in config['anchor_parameters']
    #     assert 'ratios' in config['anchor_parameters']
    #     assert 'scales' in config['anchor_parameters']
    #     assert config['anchor_parameters']['sizes'] == '32 64 128 256 512'
    #     assert config['anchor_parameters']['strides'] == '8 16 32 64 128'
    #     assert config['anchor_parameters']['ratios'] == '0.5 1 2 3'
    #     assert config['anchor_parameters']['scales'] == '1 1.2 1.6'

    def create_anchor_params_config(self):
        config = configparser.ConfigParser()
        config['anchor_parameters'] = {}
        config['anchor_parameters']['sizes'] = '32 64 128 256 512'
        config['anchor_parameters']['strides'] = '8 16 32 64 128'
        config['anchor_parameters']['ratios'] = '0.5 1'
        config['anchor_parameters']['scales'] = '1 1.2 1.6'
        return config

    # def test_parse_anchor_parameters(self):
    #     config = self.create_anchor_params_config()
    #     anchor_params_parsed = parse_anchor_parameters(config)

    #     sizes = [32, 64, 128, 256, 512]
    #     strides = [8, 16, 32, 64, 128]
    #     ratios = np.array([0.5, 1], K.floatx())
    #     scales = np.array([1, 1.2, 1.6], K.floatx())

    #     self.assertEqual(sizes, anchor_params_parsed.sizes)
    #     self.assertEqual(strides, anchor_params_parsed.strides)
    #     self.assertAllEqual(ratios, anchor_params_parsed.ratios)
    #     self.assertAllEqual(scales, anchor_params_parsed.scales)

    def test_anchors_for_shape_dimensions(self):
        sizes = [32, 64, 128]
        strides = [8, 16, 32]
        ratios = np.array([0.5, 1, 2, 3], K.floatx())
        scales = np.array([1, 1.2, 1.6], K.floatx())
        anchor_params = AnchorParameters(sizes, strides, ratios, scales)

        pyramid_levels = [3, 4, 5]
        image_shape = (64, 64)
        all_anchors = anchors_for_shape(
            image_shape,
            pyramid_levels=pyramid_levels,
            anchor_params=anchor_params)

        self.assertTupleEqual(all_anchors.shape, (1008, 4))

    def test_anchors_for_shape_values(self):
        sizes = [12]
        strides = [8]
        ratios = np.array([1, 2], K.floatx())
        scales = np.array([1, 2], K.floatx())
        anchor_params = AnchorParameters(sizes, strides, ratios, scales)

        pyramid_levels = [3]
        image_shape = (16, 16)
        all_anchors = anchors_for_shape(
            image_shape,
            pyramid_levels=pyramid_levels,
            anchor_params=anchor_params)

        # using almost_equal for floating point imprecisions
        self.assertAllClose(all_anchors[0, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[1, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[2, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[3, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[4, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[5, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[6, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[7, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[8, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[9, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[10, :], [
            strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[11, :], [
            strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[12, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[13, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        ])
        self.assertAllClose(all_anchors[14, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        ])
        self.assertAllClose(all_anchors[15, :], [
            strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
            strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        ])


if __name__ == '__main__':
    test.main()
