"""
normalization_layers.py

Layers to noramlize input images for 2D and 3D images

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


class ImageNormalization2D(Layer):
    def __init__(self, norm_method='std', filter_size=61, data_format=None, **kwargs):
        super(ImageNormalization2D, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = K.image_data_format() if data_format is None else data_format

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3 # hardcoded for 2D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def compute_output_shape(self, input_shape):
        return input_shape

    def _average_filter(self, inputs):
        in_channels = inputs.shape[self.channel_axis]
        W = np.ones((self.filter_size, self.filter_size, in_channels, 1))

        W /= W.size
        kernel = tf.Variable(W.astype(K.floatx()))

        data_format = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
        outputs = tf.nn.depthwise_conv2d(inputs, kernel, [1, 1, 1, 1],
                                         padding='SAME', data_format=data_format)
        return outputs

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(tf.square(inputs))
        output = tf.sqrt(c2 - c1 * c1) + epsilon
        return output

    def _reduce_median(self, inputs, axes=None):
        # TODO: top_k cannot take None as batch dimension, and tf.rank cannot be iterated
        input_shape = tf.shape(inputs)
        rank = tf.rank(inputs)

        new_shape = [input_shape[axis] for axis in range(rank) if axis not in axes]
        new_shape.append(-1)

        reshaped_inputs = tf.reshape(inputs, new_shape)
        median_index = reshaped_inputs.get_shape()[-1] // 2

        median = tf.nn.top_k(reshaped_inputs, k=median_index)
        return median

    def call(self, inputs):
        if not self.norm_method:
            outputs = inputs

        elif self.norm_method == 'std':
            outputs = inputs - self._average_filter(inputs)
            outputs /= self._window_std_filter(outputs)

        elif self.norm_method == 'max':
            outputs = inputs / tf.reduce_max(inputs)
            outputs -= self._average_filter(outputs)

        elif self.norm_method == 'median':
            reduce_axes = list(range(len(inputs.shape)))[1:]
            reduce_axes.remove(self.channel_axis)
            # mean = self._reduce_median(inputs, axes=reduce_axes)
            mean = tf.contrib.distributions.percentile(inputs, 50.)
            outputs = inputs / mean
            outputs -= self._average_filter(outputs)
        else:
            raise NotImplementedError('"{}" is not a valid norm_method'.format(self.norm_method))

        return outputs

    def get_config(self):
        config = {
            'norm_method': self.norm_method,
            'filter_size': self.filter_size,
            'data_format': self.data_format
        }
        base_config = super(ImageNormalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ImageNormalization3D(Layer):
    def __init__(self, norm_method='std', filter_size=61, data_format=None, **kwargs):
        super(ImageNormalization3D, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = K.image_data_format() if data_format is None else data_format

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 4 # hardcoded for 3D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def compute_output_shape(self, input_shape):
        return input_shape

    def _average_filter(self, inputs):
        in_channels = inputs.shape[self.channel_axis]
        depth = inputs.shape[2 if self.data_format == 'channels_first' else 1]
        W = np.ones((depth, self.filter_size, self.filter_size, in_channels, 1))

        W /= W.size
        kernel = tf.Variable(W.astype(K.floatx()))

        data_format = 'NCDHW' if self.data_format == 'channels_first' else 'NDHWC'
        # TODO: conv3d vs depthwise_conv2d?
        outputs = tf.nn.conv3d(inputs, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=data_format)
        return outputs

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(tf.square(inputs))
        output = tf.sqrt(c2 - c1 * c1) + epsilon
        return output

    def _reduce_median(self, inputs, axes=None):
        # TODO: top_k cannot take None as batch dimension, and tf.rank cannot be iterated
        input_shape = tf.shape(inputs)
        rank = tf.rank(inputs)

        new_shape = [input_shape[axis] for axis in range(rank) if axis not in axes]
        new_shape.append(-1)

        reshaped_inputs = tf.reshape(inputs, new_shape)
        median_index = reshaped_inputs.get_shape()[-1] // 2

        median = tf.nn.top_k(reshaped_inputs, k=median_index)
        return median

    def call(self, inputs):
        if not self.norm_method:
            outputs = inputs

        elif self.norm_method == 'whole_image':
            reduce_axes = [3, 4] if self.data_format == 'channels_first' else [2, 3]
            outputs = inputs - tf.reduce_mean(inputs, axis=reduce_axes, keepdims=True)
            outputs /= K.std(inputs, axis=reduce_axes, keepdims=True)

        elif self.norm_method == 'std':
            outputs = inputs - self._average_filter(inputs)
            outputs /= self._window_std_filter(outputs)

        elif self.norm_method == 'max':
            outputs = inputs / tf.reduce_max(inputs)
            outputs -= self._average_filter(outputs)

        elif self.norm_method == 'median':
            reduce_axes = list(range(len(inputs.shape)))[1:]
            reduce_axes.remove(self.channel_axis)
            # mean = self._reduce_median(inputs, axes=reduce_axes)
            mean = tf.contrib.distributions.percentile(inputs, 50.)
            outputs = inputs / mean
            outputs -= self._average_filter(outputs)
        else:
            raise NotImplementedError('"{}" is not a valid norm_method'.format(self.norm_method))

        return outputs

    def get_config(self):
        config = {
            'norm_method': self.norm_method,
            'filter_size': self.filter_size,
            'data_format': self.data_format
        }
        base_config = super(ImageNormalization3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
