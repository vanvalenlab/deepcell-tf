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
"""Tests for the upsampling layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell.utils import testing_utils
from deepcell import layers


class TestUpsampleLike(test.TestCase):

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_simple(self):
        # channels_last
        with self.cached_session():
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
        with self.cached_session():
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

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_simple_3d(self):
        with self.cached_session():
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
        with self.cached_session():
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

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_mini_batch(self):
        with self.cached_session():
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


class TestUpsample(test.TestCase):
    @tf_test_util.run_in_graph_and_eager_modes()
    def test_simple(self):
        with self.cached_session():
            testing_utils.layer_test(
                layers.Upsample,
                kwargs={'target_size': (2, 2)},
                custom_objects={'Upsample': layers.Upsample},
                input_shape=(3, 5, 6, 4))
            testing_utils.layer_test(
                layers.Upsample,
                kwargs={'target_size': (2, 2),
                        'data_format': 'channels_first'},
                custom_objects={'Upsample': layers.Upsample},
                input_shape=(3, 4, 5, 6))
