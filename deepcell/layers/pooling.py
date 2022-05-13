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
"""Layers to encode location data"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils


class DilatedMaxPool2D(Layer):
    """Dilated max pooling layer for 2D inputs (e.g. images).

    Args:
        pool_size (int): An integer or tuple/list of 2 integers:
            (pool_height, pool_width) specifying the size of the pooling
            window. Can be a single integer to specify the same value for
            all spatial dimensions.
        strides (int): An integer or tuple/list of 2 integers,
            specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        dilation_rate (int): An integer or tuple/list of 2 integers,
            specifying the dilation rate for the pooling.
        padding (str): The padding method, either ``"valid"`` or ``"same"``
            (case-insensitive).
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self, pool_size=(2, 2), strides=None, dilation_rate=1,
                 padding='valid', data_format=None, **kwargs):
        super(DilatedMaxPool2D, self).__init__(**kwargs)
        if strides is None or dilation_rate != 1 and dilation_rate != (1, 1):
            strides = (1, 1)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]

        # TODO: workaround! padding = 'same' shapes do not match
        _padding = self.padding
        self.padding = 'valid'

        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             padding=self.padding,
                                             stride=self.strides[0],
                                             dilation=self.dilation_rate[0])

        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             padding=self.padding,
                                             stride=self.strides[1],
                                             dilation=self.dilation_rate[1])

        # END workaround
        self.padding = _padding

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], rows, cols)
        else:
            output_shape = (input_shape[0], rows, cols, input_shape[3])

        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 1])

        if self.padding == 'valid':
            outputs = tf.nn.pool(inputs,
                                 window_shape=self.pool_size,
                                 pooling_type='MAX',
                                 padding=self.padding.upper(),
                                 dilations=self.dilation_rate,
                                 strides=self.strides,
                                 data_format='NHWC')

        elif self.padding == 'same':
            input_shape = K.int_shape(inputs)
            rows = input_shape[1]
            cols = input_shape[2]

            rows_unpadded = conv_utils.conv_output_length(
                rows, self.pool_size[0],
                padding='valid',
                stride=self.strides[0],
                dilation=self.dilation_rate[0])

            cols_unpadded = conv_utils.conv_output_length(
                cols, self.pool_size[1],
                padding='valid',
                stride=self.strides[1],
                dilation=self.dilation_rate[1])

            w_pad = (rows - rows_unpadded) // 2
            h_pad = (cols - cols_unpadded) // 2

            w_pad = (w_pad, w_pad)
            h_pad = (h_pad, h_pad)

            pattern = [[0, 0], list(w_pad), list(h_pad), [0, 0]]

            # Pad the image
            outputs = tf.pad(inputs, pattern, mode='REFLECT')

            # Perform pooling
            outputs = tf.nn.pool(inputs,
                                 window_shape=self.pool_size,
                                 pooling_type='MAX',
                                 padding='VALID',
                                 dilations=self.dilation_rate,
                                 strides=self.strides,
                                 data_format='NHWC')

        if self.data_format == 'channels_first':
            outputs = K.permute_dimensions(outputs, pattern=[0, 3, 1, 2])

        return outputs

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'strides': self.strides,
            'data_format': self.data_format
        }
        base_config = super(DilatedMaxPool2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DilatedMaxPool3D(Layer):
    """Dilated max pooling layer for 3D inputs.

    Args:
        pool_size (int): An integer or tuple/list of 2 integers:
            (pool_height, pool_width) specifying the size of the pooling
            window. Can be a single integer to specify the same value for
            all spatial dimensions.
        strides (int): An integer or tuple/list of 2 integers,
            specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        dilation_rate (int): An integer or tuple/list of 2 integers,
            specifying the dilation rate for the pooling.
        padding (str): The padding method, either ``"valid"`` or ``"same"``
            (case-insensitive).
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """
    def __init__(self, pool_size=(1, 2, 2), strides=None, dilation_rate=1,
                 padding='valid', data_format=None, **kwargs):
        super(DilatedMaxPool3D, self).__init__(**kwargs)
        data_format = conv_utils.normalize_data_format(data_format)
        if strides is None or dilation_rate != 1 and dilation_rate != (1, 1, 1):
            strides = (1, 1, 1)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 3,
                                                        'dilation_rate')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            time = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
        else:
            time = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]

        # TODO: workaround! padding = 'same' shapes do not match
        _padding = self.padding
        self.padding = 'valid'

        time = conv_utils.conv_output_length(time, self.pool_size[0],
                                             padding=self.padding,
                                             stride=self.strides[0],
                                             dilation=self.dilation_rate[0])

        rows = conv_utils.conv_output_length(rows, self.pool_size[1],
                                             padding=self.padding,
                                             stride=self.strides[1],
                                             dilation=self.dilation_rate[1])

        cols = conv_utils.conv_output_length(cols, self.pool_size[2],
                                             padding=self.padding,
                                             stride=self.strides[2],
                                             dilation=self.dilation_rate[2])

        # END workaround
        self.padding = _padding

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], time, rows, cols)
        else:
            output_shape = (input_shape[0], time, rows, cols, input_shape[4])

        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 4, 1])

        padding_input = self.padding.upper()

        if self.padding == 'valid':
            outputs = tf.nn.pool(inputs,
                                 window_shape=self.pool_size,
                                 pooling_type='MAX',
                                 padding=padding_input,
                                 dilations=self.dilation_rate,
                                 strides=self.strides,
                                 data_format='NDHWC')
        elif self.padding == 'same':
            input_shape = K.int_shape(inputs)
            times = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]

            times_unpadded = conv_utils.conv_output_length(
                times, self.pool_size[0],
                padding='valid',
                stride=self.strides[0],
                dilation=self.dilation_rate[0])

            rows_unpadded = conv_utils.conv_output_length(
                rows, self.pool_size[1],
                padding='valid',
                stride=self.strides[0],
                dilation=self.dilation_rate[1])

            cols_unpadded = conv_utils.conv_output_length(
                cols, self.pool_size[2],
                padding='valid',
                stride=self.strides[1],
                dilation=self.dilation_rate[2])

            t_pad = (times - times_unpadded) // 2
            w_pad = (rows - rows_unpadded) // 2
            h_pad = (cols - cols_unpadded) // 2

            t_pad = (t_pad, t_pad)
            w_pad = (w_pad, w_pad)
            h_pad = (h_pad, h_pad)

            pattern = [[0, 0], list(t_pad), list(w_pad), list(h_pad), [0, 0]]

            # Pad the image
            outputs = tf.pad(inputs, pattern, mode='REFLECT')

            # Perform pooling
            outputs = tf.nn.pool(inputs,
                                 window_shape=self.pool_size,
                                 pooling_type='MAX',
                                 padding='VALID',
                                 dilations=self.dilation_rate,
                                 strides=self.strides,
                                 data_format='NDHWC')

        if self.data_format == 'channels_first':
            outputs = K.permute_dimensions(outputs, pattern=[0, 4, 1, 2, 3])

        return outputs

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'strides': self.strides,
            'data_format': self.data_format
        }
        base_config = super(DilatedMaxPool3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
