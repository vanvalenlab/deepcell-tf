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
"""Tests for the upsampling layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.platform import test

from deepcell import layers


@keras_parameterized.run_all_keras_modes
class TestUpsampleLike(keras_parameterized.TestCase):

    def test_simple(self):
        # channels_last
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike()

        # create input source
        source = np.zeros((1, 2, 2, 1), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 5, 5, 1), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])

        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)
        # channels_first
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike(
            data_format='channels_first')

        # create input source
        source = np.zeros((1, 1, 2, 2), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 1, 5, 5), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])
        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)

    def test_simple_3d(self):
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike()

        # create input source
        source = np.zeros((1, 2, 2, 2, 1), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 5, 5, 5, 1), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])

        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)

        # channels_first
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike(
            data_format='channels_first')

        # create input source
        source = np.zeros((1, 1, 2, 2, 2), dtype=K.floatx())
        source = K.variable(source)
        target = np.zeros((1, 1, 5, 5, 5), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = upsample_like_layer.compute_output_shape(
            [source.shape, target.shape])
        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)

    def test_mini_batch(self):
        # create simple UpsampleLike layer
        upsample_like_layer = layers.UpsampleLike()

        # create input source
        source = np.zeros((2, 2, 2, 1), dtype=K.floatx())
        source = K.variable(source)

        target = np.zeros((2, 5, 5, 1), dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        actual = upsample_like_layer.call([source, target])
        actual = K.get_value(actual)

        self.assertAllEqual(actual, expected)
