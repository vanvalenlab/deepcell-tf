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
"""Layers to encode location data"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import InputSpec
try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils


class DilatedMaxPool2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, dilation_rate=1,
                 padding='valid', data_format=None, **kwargs):
        super(DilatedMaxPool2D, self).__init__(**kwargs)
        if dilation_rate != 1:
            strides = (1, 1)
        elif strides is None:
            strides = (1, 1)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.dilation_rate = dilation_rate
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

        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             padding=self.padding,
                                             stride=self.strides[0],
                                             dilation=self.dilation_rate)

        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             padding=self.padding,
                                             stride=self.strides[1],
                                             dilation=self.dilation_rate)

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], rows, cols)
        else:
            output_shape = (input_shape[0], rows, cols, input_shape[3])

        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 1])

        padding_input = self.padding.upper()
        dilation_rate = conv_utils.normalize_tuple(
            self.dilation_rate, 2, 'dilation_rate')

        outputs = tf.nn.pool(inputs,
                             window_shape=self.pool_size,
                             pooling_type='MAX',
                             padding=padding_input,
                             dilation_rate=dilation_rate,
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
    def __init__(self, pool_size=(2, 2), strides=None, dilation_rate=1,
                 padding='valid', data_format=None, **kwargs):
        super(DilatedMaxPool3D, self).__init__(**kwargs)
        data_format = conv_utils.normalize_data_format(data_format)
        if dilation_rate != 1:
            strides = (1, 1, 1)
        elif strides is None:
            strides = (1, 1, 1)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 3, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
        self.dilation_rate = dilation_rate
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

        time = conv_utils.conv_output_length(time, self.pool_size[0],
                                             padding=self.padding,
                                             stride=self.strides[0],
                                             dilation=self.dilation_rate)

        rows = conv_utils.conv_output_length(rows, self.pool_size[1],
                                             padding=self.padding,
                                             stride=self.strides[1],
                                             dilation=self.dilation_rate)

        cols = conv_utils.conv_output_length(cols, self.pool_size[2],
                                             padding=self.padding,
                                             stride=self.strides[2],
                                             dilation=self.dilation_rate)

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], time, rows, cols)
        else:
            output_shape = (input_shape[0], time, rows, cols, input_shape[4])

        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 4, 1])

        padding_input = self.padding.upper()
        dilation_rate = conv_utils.normalize_tuple(
            self.dilation_rate, 3, 'dilation_rate')

        outputs = tf.nn.pool(inputs,
                             window_shape=self.pool_size,
                             pooling_type='MAX',
                             padding=padding_input,
                             dilation_rate=dilation_rate,
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
