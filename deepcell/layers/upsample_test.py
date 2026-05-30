"""Tests for the upsampling layers"""

import numpy as np
from tensorflow.keras import backend as K
from keras import keras_parameterized

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
