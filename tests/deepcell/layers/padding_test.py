# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Tests for padding layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

from deepcell import layers


def _get_random_padding(dim):
    R = lambda: np.random.randint(low=0, high=9)
    return tuple([(R(), R()) for _ in range(dim)])


class ReflectionPaddingTest(test.TestCase):

    # @tf_test_util.run_in_graph_and_eager_modes()
    # def test_reflection_padding_2d(self):
    #     num_samples = 2
    #     stack_size = 2
    #     input_num_row = 4
    #     input_num_col = 5
    #     for data_format in ['channels_first', 'channels_last']:
    #         inputs = np.ones((num_samples, input_num_row, input_num_col, stack_size))
    #         inputs = np.ones((num_samples, stack_size, input_num_row, input_num_col))

    #         # basic test
    #         with self.test_session(use_gpu=True):
    #             testing_utils.layer_test(
    #                 layers.ReflectionPadding2D,
    #                 kwargs={'padding': (2, 2),
    #                         'data_format': data_format},
    #                 input_shape=inputs.shape)
    #             testing_utils.layer_test(
    #                 layers.ReflectionPadding2D,
    #                 kwargs={'padding': ((1, 2), (3, 4)),
    #                         'data_format': data_format},
    #                 input_shape=inputs.shape)

    #         # correctness test
    #         with self.test_session(use_gpu=True):
    #             layer = layers.ReflectionPadding2D(
    #                 padding=(2, 2), data_format=data_format)
    #             layer.build(inputs.shape)
    #             output = layer(keras.backend.variable(inputs))
    #             if context.executing_eagerly():
    #                 np_output = output.numpy()
    #             else:
    #                 np_output = keras.backend.eval(output)
    #             if data_format == 'channels_last':
    #                 for offset in [0, 1, -1, -2]:
    #                     np.testing.assert_allclose(np_output[:, offset, :, :], 0.)
    #                     np.testing.assert_allclose(np_output[:, :, offset, :], 0.)
    #                 np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)
    #             elif data_format == 'channels_first':
    #                 for offset in [0, 1, -1, -2]:
    #                     np.testing.assert_allclose(np_output[:, :, offset, :], 0.)
    #                     np.testing.assert_allclose(np_output[:, :, :, offset], 0.)
    #                 np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)

    #             layer = layers.ReflectionPadding2D(
    #                 padding=((1, 2), (3, 4)), data_format=data_format)
    #             layer.build(inputs.shape)
    #             output = layer(keras.backend.variable(inputs))
    #             if context.executing_eagerly():
    #                 np_output = output.numpy()
    #             else:
    #                 np_output = keras.backend.eval(output)
    #             if data_format == 'channels_last':
    #                 for top_offset in [0]:
    #                     np.testing.assert_allclose(np_output[:, top_offset, :, :], 0.)
    #                 for bottom_offset in [-1, -2]:
    #                     np.testing.assert_allclose(np_output[:, bottom_offset, :, :], 0.)
    #                 for left_offset in [0, 1, 2]:
    #                     np.testing.assert_allclose(np_output[:, :, left_offset, :], 0.)
    #                 for right_offset in [-1, -2, -3, -4]:
    #                     np.testing.assert_allclose(np_output[:, :, right_offset, :], 0.)
    #                 np.testing.assert_allclose(np_output[:, 1:-2, 3:-4, :], 1.)
    #             elif data_format == 'channels_first':
    #                 for top_offset in [0]:
    #                     np.testing.assert_allclose(np_output[:, :, top_offset, :], 0.)
    #                 for bottom_offset in [-1, -2]:
    #                     np.testing.assert_allclose(np_output[:, :, bottom_offset, :], 0.)
    #                 for left_offset in [0, 1, 2]:
    #                     np.testing.assert_allclose(np_output[:, :, :, left_offset], 0.)
    #                 for right_offset in [-1, -2, -3, -4]:
    #                     np.testing.assert_allclose(np_output[:, :, :, right_offset], 0.)
    #                 np.testing.assert_allclose(np_output[:, :, 1:-2, 3:-4], 1.)

    #         # test incorrect use
    #         with self.assertRaises(ValueError):
    #             layers.ReflectionPadding2D(padding=(1, 1, 1))
    #         with self.assertRaises(ValueError):
    #             layers.ReflectionPadding2D(padding=None)

    # @tf_test_util.run_in_graph_and_eager_modes()
    # def test_reflection_padding_3d(self):
    #     num_samples = 2
    #     stack_size = 2
    #     input_len_dim1 = 4
    #     input_len_dim2 = 5
    #     input_len_dim3 = 3

    #     inputs = np.ones((num_samples, input_len_dim1, input_len_dim2,
    #                       input_len_dim3, stack_size))

    #     # basic test
    #     with self.test_session(use_gpu=True):
    #         testing_utils.layer_test(
    #             layers.ReflectionPadding3D,
    #             kwargs={'padding': (2, 2, 2)},
    #             input_shape=inputs.shape)

    #     # correctness test
    #     with self.test_session(use_gpu=True):
    #         layer = layers.ReflectionPadding3D(padding=(2, 2, 2))
    #         layer.build(inputs.shape)
    #         output = layer(keras.backend.variable(inputs))
    #         if context.executing_eagerly():
    #             np_output = output.numpy()
    #         else:
    #             np_output = keras.backend.eval(output)
    #         for offset in [0, 1, -1, -2]:
    #             np.testing.assert_allclose(np_output[:, offset, :, :, :], 0.)
    #             np.testing.assert_allclose(np_output[:, :, offset, :, :], 0.)
    #             np.testing.assert_allclose(np_output[:, :, :, offset, :], 0.)
    #         np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, 2:-2, :], 1.)

    #     # test incorrect use
    #     with self.assertRaises(ValueError):
    #         layers.ReflectionPadding3D(padding=(1, 1))
    #     with self.assertRaises(ValueError):
    #         layers.ReflectionPadding3D(padding=None)

    def test_reflect_padding_2d(self):
        with self.test_session():
            x, y, c = 4, 4, 1
            i1 = array_ops.placeholder(shape=(None, x, y, c), dtype='float32')
            pad = _get_random_padding(2)
            layer = layers.ReflectionPadding2D(padding=pad)
            o = layer(i1)
            expected = [None, x + pad[0][0] + pad[0][1],
                        y + pad[1][0] + pad[1][1], c]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_padding_2d_channels_first(self):
        with self.test_session():
            x, y, c = 4, 4, 1
            i1 = array_ops.placeholder(shape=(None, c, x, y), dtype='float32')
            pad = _get_random_padding(2)
            layer = layers.ReflectionPadding2D(
                padding=pad, data_format='channels_first')
            o = layer(i1)
            expected = [None, c, x + pad[0][0] + pad[0][1],
                        y + pad[1][0] + pad[1][1]]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_padding_2d_tuple_input(self):
        with self.test_session():
            x, y, c = 4, 4, 1
            i1 = array_ops.placeholder(shape=(None, x, y, c), dtype='float32')
            pad = _get_random_padding(1)[0]
            layer = layers.ReflectionPadding2D(padding=pad)
            o = layer(i1)
            expected = [None, x + 2 * pad[0], y + 2 * pad[1], c]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_padding_2d_int_input(self):
        with self.test_session():
            x, y, c = 4, 4, 1
            i1 = array_ops.placeholder(shape=(None, x, y, c), dtype='float32')
            pad = np.random.randint(low=0, high=9)
            layer = layers.ReflectionPadding2D(padding=pad)
            o = layer(i1)
            expected = [None, x + 2 * pad, y + 2 * pad, c]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_padding_2d_bad_input_size(self):
        with self.test_session():
            pad = np.random.randint(low=0, high=9)
            with self.assertRaises(ValueError):
                layers.ReflectionPadding2D(padding=(pad,))

    def test_reflect_padding_2d_bad_input_type(self):
        with self.test_session():
            with self.assertRaises(ValueError):
                layers.ReflectionPadding2D(padding=None)

class ReflectionPadding3DTest(test.TestCase):

    def test_reflect_padding_3d(self):
        with self.test_session():
            z, x, y, c = 3, 4, 4, 1
            input_shape = (None, z, x, y, c)
            i1 = array_ops.placeholder(shape=input_shape, dtype='float32')
            pad = _get_random_padding(3)
            layer = layers.ReflectionPadding3D(padding=pad)
            o = layer(i1)
            expected = [None, z + pad[0][0] + pad[0][1],
                        x + pad[1][0] + pad[1][1],
                        y + pad[2][0] + pad[2][1], c]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_padding_3d_channels_first(self):
        with self.test_session():
            z, x, y, c = 3, 4, 4, 1
            input_shape = (None, c, z, x, y)
            i1 = array_ops.placeholder(shape=input_shape, dtype='float32')
            pad = _get_random_padding(3)
            layer = layers.ReflectionPadding3D(
                padding=pad, data_format='channels_first')
            o = layer(i1)
            expected = [None, c, z + pad[0][0] + pad[0][1],
                        x + pad[1][0] + pad[1][1],
                        y + pad[2][0] + pad[2][1]]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_z_padding_3d_tuple_input(self):
        with self.test_session():
            z, x, y, c = 3, 4, 4, 1
            input_shape = (None, z, x, y, c)
            i1 = array_ops.placeholder(shape=input_shape, dtype='float32')
            padz = np.random.randint(low=0, high=9)
            padx = np.random.randint(low=0, high=9)
            pady = np.random.randint(low=0, high=9)
            layer = layers.ReflectionPadding3D(padding=(padz, padx, pady))
            o = layer(i1)
            expected = [None, z + 2 * padz, x + 2 * padx, y + 2 * pady, c]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_z_padding_3d_int_input(self):
        with self.test_session():
            z, x, y, c = 3, 4, 4, 1
            input_shape = (None, z, x, y, c)
            i1 = array_ops.placeholder(shape=input_shape, dtype='float32')
            pad = np.random.randint(low=0, high=9)
            layer = layers.ReflectionPadding3D(padding=pad)
            o = layer(i1)
            expected = [None, z + 2 * pad, x + 2 * pad, y + 2 * pad, c]
            self.assertListEqual(o.get_shape().as_list(), expected)

    def test_reflect_padding_3d_bad_input_size(self):
        with self.test_session():
            pad = np.random.randint(low=0, high=9)
            with self.assertRaises(ValueError):
                layers.ReflectionPadding3D(padding=(pad,))

    def test_reflect_padding_3d_bad_input_type(self):
        with self.test_session():
            with self.assertRaises(ValueError):
                layers.ReflectionPadding3D(padding=None)

if __name__ == '__main__':
    test.main()
