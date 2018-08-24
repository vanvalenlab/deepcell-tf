"""
location_layers.py

Layers to encode location data

@author: David Van Valen
"""
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
        data_format = conv_utils.normalize_data_format(data_format)
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

        rows = conv_utils.conv_output_length(rows, self.pool_size[0], padding=self.padding,
                                             stride=self.strides[0], dilation=self.dilation_rate)

        cols = conv_utils.conv_output_length(cols, self.pool_size[1], padding=self.padding,
                                             stride=self.strides[1], dilation=self.dilation_rate)

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], rows, cols)
        else:
            output_shape = (input_shape[0], rows, cols, input_shape[3])

        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        # dilated pooling for tensorflow backend
        if K.backend() == 'theano':
            Exception('This version of DeepCell only works with the tensorflow backend')

        df = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'

        padding_input = self.padding.upper()
        if not isinstance(self.dilation_rate, tuple):
            dilation_rate = (self.dilation_rate, self.dilation_rate)
        else:
            dilation_rate = self.dilation_rate

        output = tf.nn.pool(inputs, window_shape=self.pool_size, pooling_type='MAX',
                            padding=padding_input, dilation_rate=dilation_rate,
                            strides=self.strides, data_format=df)

        return output

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

        time = conv_utils.conv_output_length(time, self.pool_size[0], padding=self.padding,
                                             stride=self.strides[0], dilation=self.dilation_rate)

        rows = conv_utils.conv_output_length(rows, self.pool_size[1], padding=self.padding,
                                             stride=self.strides[1], dilation=self.dilation_rate)

        cols = conv_utils.conv_output_length(cols, self.pool_size[2], padding=self.padding,
                                             stride=self.strides[2], dilation=self.dilation_rate)

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], time, rows, cols)
        else:
            output_shape = (input_shape[0], time, rows, cols, input_shape[4])

        return output_shape

    def call(self, inputs):
        # dilated pooling for tensorflow backend
        if K.backend() == 'theano':
            Exception('This version of DeepCell only works with the tensorflow backend')

        df = 'NCDHW' if self.data_format == 'channels_first' else 'NDHWC'

        padding_input = self.padding.upper()
        if not isinstance(self.dilation_rate, tuple):
            dilation_rate = (self.dilation_rate, self.dilation_rate, self.dilation_rate)
        else:
            dilation_rate = self.dilation_rate

        output = tf.nn.pool(inputs, window_shape=self.pool_size, pooling_type='MAX',
                            padding=padding_input, dilation_rate=dilation_rate,
                            strides=self.strides, data_format=df)

        return output

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
