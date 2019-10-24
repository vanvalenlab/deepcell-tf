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
"""Tests for the pooling layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell.utils import testing_utils
from deepcell import layers


class DilatedMaxPoolingTest(test.TestCase):

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_dilated_max_pool_2d(self):
        pool_size = (3, 3)
        custom_objects = {'DilatedMaxPool2D': layers.DilatedMaxPool2D}
        for strides in [(1, 1), (2, 2), None]:
            for dilation_rate in [1, 2, (1, 2)]:
                for padding in ['valid', 'same']:
                    with self.test_session():
                        testing_utils.layer_test(
                            layers.DilatedMaxPool2D,
                            kwargs={'strides': strides,
                                    'pool_size': pool_size,
                                    'padding': padding,
                                    'dilation_rate': dilation_rate,
                                    'data_format': 'channels_last'},
                            custom_objects=custom_objects,
                            input_shape=(3, 5, 6, 4))
                        testing_utils.layer_test(
                            layers.DilatedMaxPool2D,
                            kwargs={'strides': strides,
                                    'pool_size': pool_size,
                                    'padding': padding,
                                    'dilation_rate': dilation_rate,
                                    'data_format': 'channels_first'},
                            custom_objects=custom_objects,
                            input_shape=(3, 4, 5, 6))

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_dilated_max_pool_3d(self):
        custom_objects = {'DilatedMaxPool3D': layers.DilatedMaxPool3D}
        pool_size = (3, 3, 3)
        for strides in [1, 2, None]:
            for dilation_rate in [1, 2, (1, 2, 2)]:
                for padding in ['valid', 'same']:
                    with self.test_session():
                        testing_utils.layer_test(
                            layers.DilatedMaxPool3D,
                            kwargs={'strides': strides,
                                    'padding': padding,
                                    'dilation_rate': dilation_rate,
                                    'pool_size': pool_size},
                            custom_objects=custom_objects,
                            input_shape=(3, 11, 12, 10, 4))
                        testing_utils.layer_test(
                            layers.DilatedMaxPool3D,
                            kwargs={'strides': strides,
                                    'padding': padding,
                                    'dilation_rate': dilation_rate,
                                    'data_format': 'channels_first',
                                    'pool_size': pool_size},
                            custom_objects=custom_objects,
                            input_shape=(3, 4, 11, 12, 10))
