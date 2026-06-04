"""Tests for padding layers."""


import functools

import numpy as np
import tensorflow as tf

# from tensorflow.python.eager import context
from keras import keras_parameterized
from keras import testing_utils
from tensorflow.python.platform import test

from deepcell import layers


def _get_random_padding(dim):
    R = functools.partial(np.random.randint, low=0, high=9)
    return tuple([(R(), R()) for _ in range(dim)])


@keras_parameterized.run_all_keras_modes
class ReflectionPaddingTest(keras_parameterized.TestCase):

    def test_reflection_padding_2d(self):
        num_samples = 2
        stack_size = 2
        input_num_row = 4
        input_num_col = 5

        custom_objects = {'ReflectionPadding2D': layers.ReflectionPadding2D}
        ins1 = np.ones((num_samples, input_num_row, input_num_col, stack_size))
        ins2 = np.ones((num_samples, stack_size, input_num_row, input_num_col))
        data_formats = ['channels_first', 'channels_last']
        with tf.keras.utils.custom_object_scope(custom_objects):
            for data_format, inputs in zip(data_formats, [ins2, ins1]):
                # basic test
                testing_utils.layer_test(
                    layers.ReflectionPadding2D,
                    kwargs={'padding': (2, 2),
                            'data_format': data_format},
                    input_shape=inputs.shape)
                testing_utils.layer_test(
                    layers.ReflectionPadding2D,
                    kwargs={'padding': ((1, 2), (3, 4)),
                            'data_format': data_format},
                    input_shape=inputs.shape)

        # correctness test
        # with self.cached_session():
        #     layer = layers.ReflectionPadding2D(
        #         padding=(2, 2), data_format=data_format)
        #     layer.build(inputs.shape)
        #     output = layer(tf.keras.backend.variable(inputs))
        #     if context.executing_eagerly():
        #         np_output = output.numpy()
        #     else:
        #         np_output = tf.keras.backend.eval(output)
        #     if data_format == 'channels_last':
        #         for offset in [0, 1, -1, -2]:
        #             np.testing.assert_allclose(np_output[:, offset, :, :], 0.)
        #             np.testing.assert_allclose(np_output[:, :, offset, :], 0.)
        #         np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)
        #     elif data_format == 'channels_first':
        #         for offset in [0, 1, -1, -2]:
        #             np.testing.assert_allclose(np_output[:, :, offset, :], 0.)
        #             np.testing.assert_allclose(np_output[:, :, :, offset], 0.)
        #         np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)

        #     layer = layers.ReflectionPadding2D(
        #         padding=((1, 2), (3, 4)), data_format=data_format)
        #     layer.build(inputs.shape)
        #     output = layer(tf.keras.backend.variable(inputs))
        #     if context.executing_eagerly():
        #         np_output = output.numpy()
        #     else:
        #         np_output = tf.keras.backend.eval(output)
        #     if data_format == 'channels_last':
        #         for top_offset in [0]:
        #             np.testing.assert_allclose(np_output[:, top_offset, :, :], 0.)
        #         for bottom_offset in [-1, -2]:
        #             np.testing.assert_allclose(np_output[:, bottom_offset, :, :], 0.)
        #         for left_offset in [0, 1, 2]:
        #             np.testing.assert_allclose(np_output[:, :, left_offset, :], 0.)
        #         for right_offset in [-1, -2, -3, -4]:
        #             np.testing.assert_allclose(np_output[:, :, right_offset, :], 0.)
        #         np.testing.assert_allclose(np_output[:, 1:-2, 3:-4, :], 1.)
        #     elif data_format == 'channels_first':
        #         for top_offset in [0]:
        #             np.testing.assert_allclose(np_output[:, :, top_offset, :], 0.)
        #         for bottom_offset in [-1, -2]:
        #             np.testing.assert_allclose(np_output[:, :, bottom_offset, :], 0.)
        #         for left_offset in [0, 1, 2]:
        #             np.testing.assert_allclose(np_output[:, :, :, left_offset], 0.)
        #         for right_offset in [-1, -2, -3, -4]:
        #             np.testing.assert_allclose(np_output[:, :, :, right_offset], 0.)
        #         np.testing.assert_allclose(np_output[:, :, 1:-2, 3:-4], 1.)

        # test incorrect use
        with self.assertRaises(ValueError):
            layers.ReflectionPadding2D(padding=(1, 1, 1))
        with self.assertRaises(ValueError):
            layers.ReflectionPadding2D(padding=None)

    def test_reflection_padding_3d(self):
        num_samples = 2
        stack_size = 2
        input_len_dim1 = 4
        input_len_dim2 = 5
        input_len_dim3 = 3

        custom_objects = {'ReflectionPadding3D': layers.ReflectionPadding3D}
        inputs1 = np.ones((num_samples, input_len_dim1, input_len_dim2,
                           input_len_dim3, stack_size))
        inputs2 = np.ones((num_samples, stack_size, input_len_dim1,
                           input_len_dim2, input_len_dim3))
        data_formats = ['channels_first', 'channels_last']
        with tf.keras.utils.custom_object_scope(custom_objects):
            for data_format, inputs in zip(data_formats, [inputs2, inputs1]):
                # basic test
                testing_utils.layer_test(
                    layers.ReflectionPadding3D,
                    kwargs={'padding': (2, 2, 2),
                            'data_format': data_format},
                    input_shape=inputs.shape)

        # correctness test
        # with self.cached_session():
        #     layer = layers.ReflectionPadding3D(padding=(2, 2, 2))
        #     layer.build(inputs.shape)
        #     output = layer(tf.keras.backend.variable(inputs))
        #     if context.executing_eagerly():
        #         np_output = output.numpy()
        #     else:
        #         np_output = tf.keras.backend.eval(output)
        #     for offset in [0, 1, -1, -2]:
        #         np.testing.assert_allclose(np_output[:, offset, :, :, :], 0.)
        #         np.testing.assert_allclose(np_output[:, :, offset, :, :], 0.)
        #         np.testing.assert_allclose(np_output[:, :, :, offset, :], 0.)
        #     np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, 2:-2, :], 1.)

        # test incorrect use
        with self.assertRaises(ValueError):
            layers.ReflectionPadding3D(padding=(1, 1))
        with self.assertRaises(ValueError):
            layers.ReflectionPadding3D(padding=None)


if __name__ == '__main__':
    test.main()
