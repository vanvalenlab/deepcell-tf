"""Tests for the upsampling layers"""

import numpy as np
from tensorflow.keras import backend as K
from keras import keras_parameterized
from keras import testing_utils
from tensorflow.keras.utils import custom_object_scope

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
