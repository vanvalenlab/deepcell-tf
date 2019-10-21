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
"""Tests for the normalization layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.python import keras
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util as tf_test_util

from deepcell.utils import testing_utils
from deepcell import layers


class ImageNormalizationTest(test.TestCase):

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_normalize_2d(self):
        custom_objects = {'ImageNormalization2D': layers.ImageNormalization2D}
        norm_methods = [None, 'std', 'max', 'whole_image']
        with self.cached_session()():
            # test each norm method
            for norm_method in norm_methods:
                testing_utils.layer_test(
                    layers.ImageNormalization2D,
                    kwargs={'norm_method': norm_method,
                            'filter_size': 3,
                            'data_format': 'channels_last'},
                    custom_objects=custom_objects,
                    input_shape=(3, 5, 6, 4))
                testing_utils.layer_test(
                    layers.ImageNormalization2D,
                    kwargs={'norm_method': norm_method,
                            'filter_size': 3,
                            'data_format': 'channels_first'},
                    custom_objects=custom_objects,
                    input_shape=(3, 4, 5, 6))
            # test constraints and bias
            k_constraint = keras.constraints.max_norm(0.01)
            b_constraint = keras.constraints.max_norm(0.01)
            layer = layers.ImageNormalization2D(
                use_bias=True,
                kernel_constraint=k_constraint,
                bias_constraint=b_constraint)
            layer(keras.backend.variable(np.ones((3, 5, 6, 4))))
            self.assertEqual(layer.kernel.constraint, k_constraint)
            self.assertEqual(layer.bias.constraint, b_constraint)
            # test bad norm_method
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization2D(norm_method='invalid')
            # test bad input dimensions
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization2D()
                layer.build([3, 10, 11, 12, 4])
            # test invalid channel
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization2D()
                layer.build([3, 5, 6, None])

    # @tf_test_util.run_in_graph_and_eager_modes()
    def test_normalize_3d(self):
        custom_objects = {'ImageNormalization3D': layers.ImageNormalization3D}
        norm_methods = [None, 'std', 'max', 'whole_image']
        with self.cached_session()():
            # test each norm method
            for norm_method in norm_methods:
                testing_utils.layer_test(
                    layers.ImageNormalization3D,
                    kwargs={'norm_method': norm_method,
                            'filter_size': 3,
                            'data_format': 'channels_last'},
                    custom_objects=custom_objects,
                    input_shape=(3, 11, 12, 10, 4))
                testing_utils.layer_test(
                    layers.ImageNormalization3D,
                    kwargs={'norm_method': norm_method,
                            'filter_size': 3,
                            'data_format': 'channels_first'},
                    custom_objects=custom_objects,
                    input_shape=(3, 4, 11, 12, 10))
            # test constraints and bias
            k_constraint = keras.constraints.max_norm(0.01)
            b_constraint = keras.constraints.max_norm(0.01)
            layer = layers.ImageNormalization3D(
                use_bias=True,
                kernel_constraint=k_constraint,
                bias_constraint=b_constraint)
            layer(keras.backend.variable(np.ones((3, 4, 11, 12, 10))))
            self.assertEqual(layer.kernel.constraint, k_constraint)
            self.assertEqual(layer.bias.constraint, b_constraint)
            # test bad norm_method
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization3D(norm_method='invalid')
            # test bad input dimensions
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization3D()
                layer.build([3, 5, 6, 4])
            # test invalid channel
            with self.assertRaises(ValueError):
                layer = layers.ImageNormalization3D()
                layer.build([3, 10, 11, 12, None])
