"""
location_layers.py

Layers to encode location data

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import InputSpec
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

        return output_shape

    def call(self, inputs):
        # dilated pooling for tensorflow backend
        if K.backend() == 'theano':
            Exception('This version of DeepCell only works with the tensorflow backend')

        df = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'

        padding_input = self.padding.upper()
        dilation_rate = (self.dilation_rate, self.dilation_rate)

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
