# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils


class ImageNormalization2D(Layer):
    """Image Normalization layer for 2D data.

    Args:
        norm_method (str): Normalization method to use, one of:
            "std", "max", "whole_image", None.
        filter_size (int): The length of the convolution window.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        activation (function): Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: ``a(x) = x``).
        use_bias (bool): Whether the layer uses a bias.
        kernel_initializer (function): Initializer for the ``kernel`` weights
            matrix, used for the linear transformation of the inputs.
        bias_initializer (function): Initializer for the bias vector. If None,
            the default initializer will be used.
        kernel_regularizer (function): Regularizer function applied to the
            ``kernel`` weights matrix.
        bias_regularizer (function): Regularizer function applied to the
            bias vector.
        activity_regularizer (function): Regularizer function applied to.
        kernel_constraint (function): Constraint function applied to
            the ``kernel`` weights matrix.
        bias_constraint (function): Constraint function applied to the
            bias vector.
    """
    def __init__(self,
                 norm_method='std',
                 filter_size=61,
                 data_format=None,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.valid_modes = {'std', 'max', None, 'whole_image'}
        if norm_method not in self.valid_modes:
            raise ValueError('Invalid `norm_method`: "{}". '
                             'Use one of {}.'.format(
                                 norm_method, self.valid_modes))
        if 'trainable' not in kwargs:
            kwargs['trainable'] = False
        super(ImageNormalization2D, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)  # hardcoded for 2D data

        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = conv_utils.normalize_data_format(data_format)

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 3  # hardcoded for 2D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4, '
                             'received input shape: %s' % input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})

        kernel_shape = (self.filter_size, self.filter_size, input_dim, 1)
        # self.kernel = self.add_weight(
        #     name='kernel',
        #     shape=kernel_shape,
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     trainable=False,
        #     dtype=self.dtype)

        W = K.ones(kernel_shape, dtype=K.floatx())
        W = W / K.cast(K.prod(K.int_shape(W)), dtype=K.floatx())
        self.kernel = W
        # self.set_weights([W])

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filter_size, self.filter_size),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=False,
                dtype=self.dtype)
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def _average_filter(self, inputs):
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 1])
        outputs = tf.nn.depthwise_conv2d(inputs, self.kernel, [1, 1, 1, 1],
                                         padding='SAME', data_format='NHWC')
        if self.data_format == 'channels_first':
            outputs = K.permute_dimensions(outputs, pattern=[0, 3, 1, 2])
        return outputs

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(K.square(inputs))
        output = K.sqrt(c2 - c1 * c1) + epsilon
        return output

    def call(self, inputs):
        if not self.norm_method:
            outputs = inputs

        elif self.norm_method == 'whole_image':
            axes = [2, 3] if self.channel_axis == 1 else [1, 2]
            outputs = inputs - K.mean(inputs, axis=axes, keepdims=True)
            outputs = outputs / (K.std(inputs, axis=axes, keepdims=True) + K.epsilon())

        elif self.norm_method == 'std':
            outputs = inputs - self._average_filter(inputs)
            outputs = outputs / self._window_std_filter(outputs)

        elif self.norm_method == 'max':
            outputs = inputs / K.max(inputs)
            outputs = outputs - self._average_filter(outputs)

        else:
            raise NotImplementedError('"{}" is not a valid norm_method'.format(
                self.norm_method))

        return outputs

    def get_config(self):
        config = {
            'norm_method': self.norm_method,
            'filter_size': self.filter_size,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ImageNormalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ImageNormalization3D(Layer):
    """Image Normalization layer for 3D data.

    Args:
        norm_method (str): Normalization method to use, one of:
            "std", "max", "whole_image", None.
        filter_size (int): The length of the convolution window.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        activation (function): Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: ``a(x) = x``).
        use_bias (bool): Whether the layer uses a bias.
        kernel_initializer (function): Initializer for the ``kernel`` weights
            matrix, used for the linear transformation of the inputs.
        bias_initializer (function): Initializer for the bias vector. If None,
            the default initializer will be used.
        kernel_regularizer (function): Regularizer function applied to the
            ``kernel`` weights matrix.
        bias_regularizer (function): Regularizer function applied to the
            bias vector.
        activity_regularizer (function): Regularizer function applied to.
        kernel_constraint (function): Constraint function applied to
            the ``kernel`` weights matrix.
        bias_constraint (function): Constraint function applied to the
            bias vector.
    """
    def __init__(self,
                 norm_method='std',
                 filter_size=61,
                 data_format=None,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.valid_modes = {'std', 'max', None, 'whole_image'}
        if norm_method not in self.valid_modes:
            raise ValueError('Invalid `norm_method`: "{}". '
                             'Use one of {}.'.format(
                                 norm_method, self.valid_modes))
        if 'trainable' not in kwargs:
            kwargs['trainable'] = False
        super(ImageNormalization3D, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=5)  # hardcoded for 3D data

        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = conv_utils.normalize_data_format(data_format)

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = 4  # hardcoded for 3D data

        if isinstance(self.norm_method, str):
            self.norm_method = self.norm_method.lower()

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 5:
            raise ValueError('Inputs should have rank 5, '
                             'received input shape: %s' % input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined, found None: %s' % input_shape)
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})

        if self.data_format == 'channels_first':
            depth = int(input_shape[2])
        else:
            depth = int(input_shape[1])
        kernel_shape = (depth, self.filter_size, self.filter_size, input_dim, 1)

        # self.kernel = self.add_weight(
        #     'kernel',
        #     shape=kernel_shape,
        #     initializer=self.kernel_initializer,
        #     regularizer=self.kernel_regularizer,
        #     constraint=self.kernel_constraint,
        #     trainable=False,
        #     dtype=self.dtype)

        W = K.ones(kernel_shape, dtype=K.floatx())
        W = W / K.cast(K.prod(K.int_shape(W)), dtype=K.floatx())
        self.kernel = W
        # self.set_weights([W])

        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=(depth, self.filter_size, self.filter_size),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=False,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return tensor_shape.TensorShape(input_shape)

    def _average_filter(self, inputs):
        if self.data_format == 'channels_first':
            inputs = K.permute_dimensions(inputs, pattern=[0, 2, 3, 4, 1])
        # TODO: conv3d vs depthwise_conv2d?
        outputs = tf.nn.conv3d(inputs, self.kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format='NDHWC')

        if self.data_format == 'channels_first':
            outputs = K.permute_dimensions(outputs, pattern=[0, 4, 1, 2, 3])
        return outputs

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(K.square(inputs))
        output = K.sqrt(c2 - c1 * c1) + epsilon
        return output

    def call(self, inputs):
        if not self.norm_method:
            outputs = inputs

        elif self.norm_method == 'whole_image':
            axes = [3, 4] if self.channel_axis == 1 else [2, 3]
            outputs = inputs - K.mean(inputs, axis=axes, keepdims=True)
            outputs = outputs / (K.std(inputs, axis=axes, keepdims=True) + K.epsilon())

        elif self.norm_method == 'std':
            outputs = inputs - self._average_filter(inputs)
            outputs = outputs / self._window_std_filter(outputs)

        elif self.norm_method == 'max':
            outputs = inputs / K.max(inputs)
            outputs = outputs - self._average_filter(outputs)

        else:
            raise NotImplementedError('"{}" is not a valid norm_method'.format(
                self.norm_method))

        return outputs

    def get_config(self):
        config = {
            'norm_method': self.norm_method,
            'filter_size': self.filter_size,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(ImageNormalization3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
