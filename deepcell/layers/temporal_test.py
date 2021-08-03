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
class TestComparison(keras_parameterized.TestCase):
    def test_simple(self):
        # create simple Comparison layer
        comparison_layer = layers.Comparison()

        # create input before
        before_objs = 7
        before_chan = 2
        before = np.zeros((1, 5, before_objs, before_chan), dtype=K.floatx())
        before = K.variable(before)
        after_objs = 8
        after_chan = 3
        after = np.zeros((1, 5, after_objs, after_chan), dtype=K.floatx())
        after = K.variable(after)

        target_chan = before_chan + after_chan
        target_shape = tuple(list(before.shape[:-1]) + [after_objs, target_chan])
        target = np.zeros(target_shape, dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = comparison_layer.compute_output_shape(
            [before.shape, after.shape])

        actual = comparison_layer.call([before, after])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)


@keras_parameterized.run_all_keras_modes
class TestDeltaReshape(keras_parameterized.TestCase):
    def test_simple(self):
        # create simple DeltaReshape layer
        delta_reshape_layer = layers.DeltaReshape()

        # create input before
        before_objs = 7
        before = np.zeros((3, 5, before_objs, 1), dtype=K.floatx())
        before = K.variable(before)
        after_objs = 8
        after = np.zeros((1, 1, after_objs, 1), dtype=K.floatx())
        after = K.variable(after)

        target_shape = tuple(list(before.shape[:-1]) + list(after.shape[-2:]))
        target = np.zeros(target_shape, dtype=K.floatx())
        expected = target
        target = K.variable(target)

        # compute output
        computed_shape = delta_reshape_layer.compute_output_shape(
            [before.shape, after.shape])

        actual = delta_reshape_layer.call([before, after])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)
        self.assertAllEqual(actual, expected)


@keras_parameterized.run_all_keras_modes
class TestUnmerge(keras_parameterized.TestCase):

    def test_unmerge(self):
        track_length = 5
        max_cells = 10
        embedding_dim = 64

        custom_objects = {'Unmerge': layers.Unmerge}
        with self.cached_session():
            with custom_object_scope(custom_objects):
                testing_utils.layer_test(
                    layers.Unmerge,
                    kwargs={'track_length': track_length,
                            'max_cells': max_cells,
                            'embedding_dim': embedding_dim},
                    input_shape=(None, track_length * max_cells, embedding_dim))


@keras_parameterized.run_all_keras_modes
class TestTemporalMerge(keras_parameterized.TestCase):

    def test_temporal_merge(self):
        track_length = 5
        max_cells = 7
        encoder_dim = 32

        custom_objects = {'TemporalMerge': layers.TemporalMerge}
        with self.cached_session():
            with custom_object_scope(custom_objects):
                testing_utils.layer_test(
                    layers.TemporalMerge,
                    kwargs={'encoder_dim': encoder_dim},
                    input_shape=(None, track_length, max_cells, encoder_dim))
