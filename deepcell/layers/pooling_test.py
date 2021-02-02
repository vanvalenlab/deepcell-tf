# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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

from absl.testing import parameterized

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell import layers


class DilatedMaxPoolingTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            strides=[(1, 1), (2, 2), None],
            dilation_rate=[1, 2, (1, 2)],
            padding=['valid', 'same']))
    def test_dilated_max_pool_2d(self, strides, dilation_rate, padding):
        pool_size = (3, 3)
        custom_objects = {'DilatedMaxPool2D': layers.DilatedMaxPool2D}
        with self.cached_session():
            with custom_object_scope(custom_objects):
                testing_utils.layer_test(
                    layers.DilatedMaxPool2D,
                    kwargs={'strides': strides,
                            'pool_size': pool_size,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'data_format': 'channels_last'},
                    input_shape=(3, 5, 6, 4))
                testing_utils.layer_test(
                    layers.DilatedMaxPool2D,
                    kwargs={'strides': strides,
                            'pool_size': pool_size,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'data_format': 'channels_first'},
                    input_shape=(3, 4, 5, 6))

    @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            strides=[1, 2, None],
            dilation_rate=[1, 2, (1, 2, 2)],
            padding=['valid', 'same']))
    def test_dilated_max_pool_3d(self, strides, dilation_rate, padding):
        custom_objects = {'DilatedMaxPool3D': layers.DilatedMaxPool3D}
        pool_size = (3, 3, 3)
        with self.cached_session():
            with custom_object_scope(custom_objects):
                testing_utils.layer_test(
                    layers.DilatedMaxPool3D,
                    kwargs={'strides': strides,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'pool_size': pool_size},
                    input_shape=(3, 11, 12, 10, 4))
                testing_utils.layer_test(
                    layers.DilatedMaxPool3D,
                    kwargs={'strides': strides,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'data_format': 'channels_first',
                            'pool_size': pool_size},
                    input_shape=(3, 4, 11, 12, 10))


if __name__ == '__main__':
    test.main()
