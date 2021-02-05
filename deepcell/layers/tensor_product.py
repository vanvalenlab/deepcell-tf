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
"""Layers to generate tensor products for 2D and 3D data"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils


class TensorProduct(Layer):
    """Just your regular densely-connected NN layer.

    Dense implements the operation:

    ``output = activation(dot(input, kernel) + bias)``

    where ``activation`` is the element-wise activation function
    passed as the ``activation`` argument, ``kernel`` is a weights matrix
    created by the layer, and ``bias`` is a bias vector created by the layer
    (only applicable if ``use_bias`` is ``True``).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with ``kernel``.

    Args:
        output_dim (int): Positive integer, dimensionality of the output space.
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

    Input shape:
        nD tensor with shape: (batch_size, ..., input_dim).
        The most common situation would be
        a 2D input with shape (batch_size, input_dim).

    Output shape:
        nD tensor with shape: (batch_size, ..., output_dim).
        For instance, for a 2D input with shape (batch_size, input_dim),
        the output would have shape (batch_size, output_dim).
    """

    def __init__(self,
                 output_dim,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(TensorProduct, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(
                activity_regularizer), **kwargs)

        self.output_dim = int(output_dim)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`TensorProduct` should be defined. '
                             'Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(
            'kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        rank = len(inputs.get_shape().as_list())
        if self.data_format == 'channels_first':
            pattern = [0, rank - 1] + list(range(1, rank - 1))
            output = tf.tensordot(inputs, self.kernel, axes=[[1], [0]])
            output = K.permute_dimensions(output, pattern=pattern)
            # output = K.dot(inputs, self.kernel)

        elif self.data_format == 'channels_last':
            output = tf.tensordot(inputs, self.kernel, axes=[[rank - 1], [0]])

        if self.use_bias:
            output = K.bias_add(output, self.bias, self.data_format)

        if self.activation is not None:
            return self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            output_shape = tuple([input_shape[0], self.output_dim] +
                                 list(input_shape[2:]))
        else:
            output_shape = tuple(list(input_shape[:-1]) + [self.output_dim])

        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'data_format': self.data_format,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(
                self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(
                self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(
                self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(
                self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(
                self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(
                self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(
                self.bias_constraint)
        }
        base_config = super(TensorProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
