"""
normalization_layers.py

Layers to resize input images

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras._impl.keras.utils import conv_utils


class Resize(Layer):
    def __init__(self, scale=2, data_format=None, **kwargs):
        super(Resize, self).__init__(**kwargs)

        backend = K.backend()
        if backend == "theano":
            Exception('This version of DeepCell only works with the tensorflow backend')
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.scale = scale

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]

        rows *= self.scale
        cols *= self.scale

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], rows, cols)
        else:
            output_shape = (input_shape[0], rows, cols, input_shape[3])

        return output_shape

    def call(self, inputs):
        if self.data_format == 'channels_first':
            channel_last = K.permute_dimensions(inputs, (0, 2, 3, 1))
        else:
            channel_last = inputs

        input_shape = tf.shape(channel_last)

        rows = self.scale * input_shape[1]
        cols = self.scale * input_shape[2]

        resized = tf.image.resize_images(channel_last, (rows, cols))

        if self.data_format == 'channels_first':
            output = K.permute_dimensions(resized, (0, 3, 1, 2))
        else:
            output = resized

        return output

    def get_config(self):
        config = {
            'scale': self.scale,
            'data_format': self.data_format
        }
        base_config = super(Resize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
