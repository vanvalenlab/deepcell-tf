# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""Layers to noramlize input images for 2D and 3D images"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils


class ImageNormalization2D(Layer):
    def __init__(self, norm_method='std', filter_size=61, data_format=None, **kwargs):
        super(ImageNormalization2D, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = conv_utils.normalize_data_format(data_format)

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3  # hardcoded for 2D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def _average_filter(self, inputs):
        in_channels = inputs.shape[self.channel_axis]
        W = np.ones((self.filter_size, self.filter_size, in_channels, 1))

        W /= W.size
        kernel = tf.Variable(W.astype(K.floatx()))

        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])

        outputs = tf.nn.depthwise_conv2d(inputs, kernel, [1, 1, 1, 1],
                                         padding='SAME', data_format='NHWC')

        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, perm=[0, 3, 1, 2])

        return outputs

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(tf.square(inputs))
        output = tf.sqrt(c2 - c1 * c1) + epsilon
        return output

    def _reduce_median(self, inputs, axes=None):
        input_shape = tf.shape(inputs)
        rank = tf.rank(inputs)
        axes = [] if axes is None else axes

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
        self.data_format = conv_utils.normalize_data_format(data_format)

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 4  # hardcoded for 3D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def _average_filter(self, inputs):
        in_channels = inputs.shape[self.channel_axis]
        depth = inputs.shape[2 if self.data_format == 'channels_first' else 1]
        W = np.ones((depth, self.filter_size, self.filter_size, in_channels, 1))

        W /= W.size
        kernel = tf.Variable(W.astype(K.floatx()))

        # data_format = 'NCDHW' if self.data_format == 'channels_first' else 'NDHWC'
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, perm=[0, 2, 3, 4, 1])
        # TODO: conv3d vs depthwise_conv2d?
        outputs = tf.nn.conv3d(inputs, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format='NDHWC')

        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, perm=[0, 4, 1, 2, 3])

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
        axes = [] if axes is None else axes

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
