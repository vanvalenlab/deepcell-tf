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
"""Convolutational Recurrent Layers

Based on Convolutional gated recurrent networks for video segmentation
(`link <https://bit.ly/3ehmpV8>`_).

Also influenced by Keras's `GRU <https://bit.ly/3ejjBHq>`_
and  `ConvLSTM2D <https://bit.ly/2HUzJ60>`_ Layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
from tensorflow.python.keras.utils import conv_utils


class ConvGRU2DCell(DropoutRNNCellMixin, Layer):
    """Cell class for the ConvGRU2D layer.

    Args:
        filters (int): An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides (int): An integer or tuple/list of 2 integers,
            specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding (str): The padding method, either ``"valid"`` or ``"same"``
            (case-insensitive).
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        dilation_rate (int): An integer or tuple/list of 2 integers,
            specifying the dilation rate for the pooling.
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
        recurrent_initializer (function): Initializer for the
            ``recurrent_kernel`` weights matrix, used for the linear
            transformation of the recurrent state.
        bias_regularizer (function): Regularizer function applied to the
            bias vector.
        activity_regularizer (function): Regularizer function applied to.
        recurrent_regularizer (function): Regularizer function applied to
            the ``recurrent_kernel`` weights matrix.
        kernel_constraint (function): Constraint function applied to
            the ``kernel`` weights matrix.
        bias_constraint (function): Constraint function applied to the
            bias vector.
        dropout (float): Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the inputs.
        recurrent_dropout (float): Float between 0 and 1. Fraction of the
            units to drop for the linear transformation of the
            recurrent state.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(ConvGRU2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')

        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

    @property
    def state_size(self):
        return (self.filters,)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 3)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 3)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters * 3,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(
            inputs, training=training, count=3)

        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training=training, count=3)

        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z = h_tm1 * rec_dp_mask[0]
            h_tm1_r = h_tm1 * rec_dp_mask[1]
            h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
            h_tm1_z = h_tm1
            h_tm1_r = h_tm1
            h_tm1_h = h_tm1

        (kernel_z, kernel_r, kernel_h) = tf.split(self.kernel, 3, axis=3)
        (recurrent_kernel_z,
         recurrent_kernel_r,
         recurrent_kernel_h) = tf.split(self.recurrent_kernel, 3, axis=3)

        if self.use_bias:
            bias_z, bias_r, bias_h = tf.split(self.bias, 3)
        else:
            bias_z, bias_r, bias_h = None, None, None

        x_z = self.input_conv(inputs_z, kernel_z, bias_z, padding=self.padding)
        x_r = self.input_conv(inputs_r, kernel_r, bias_r, padding=self.padding)
        x_h = self.input_conv(inputs_h, kernel_h, bias_h, padding=self.padding)

        h_z = self.recurrent_conv(h_tm1_z, recurrent_kernel_z)
        h_r = self.recurrent_conv(h_tm1_r, recurrent_kernel_r)

        z = self.recurrent_activation(x_z + h_z)
        r = self.recurrent_activation(x_r + h_r)

        h_h = self.recurrent_conv(r * h_tm1_h, recurrent_kernel_h)

        # previous and candidate state mixed by update gate
        h = (1 - z) * h_tm1 + z * self.activation(x_h + h_h)

        if self.dropout + self.recurrent_dropout > 0:
            if training is None:
                h._uses_learning_phase = True

        return h, [h]

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvGRU2D(ConvRNN2D):
    """Cell class for the ConvGRU2D layer.

    Args:
        filters (int): An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides (int): An integer or tuple/list of 2 integers,
            specifying the strides of the pooling operation.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding (str): The padding method, either ``"valid"`` or ``"same"``
            (case-insensitive).
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        dilation_rate (int): An integer or tuple/list of 2 integers,
            specifying the dilation rate for the pooling.
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
        recurrent_initializer (function): Initializer for the
            ``recurrent_kernel`` weights matrix, used for the linear
            transformation of the recurrent state.
        bias_regularizer (function): Regularizer function applied to the
            bias vector.
        activity_regularizer (function): Regularizer function applied to.
        recurrent_regularizer (function): Regularizer function applied to
            the ``recurrent_kernel`` weights matrix.
        kernel_constraint (function): Constraint function applied to
            the ``kernel`` weights matrix.
        bias_constraint (function): Constraint function applied to the
            bias vector.
        return_sequences (bool): Whether to return the last output in the
            output sequence, or the full sequence.
        return_state (bool): Whether to return the last state
            in addition to the output.
        go_backwards (bool): If True, process the input sequence backwards.
        stateful (bool): If True, the last state for each sample at index i in
            a batch will be used as initial state for the sample of index i in
            the following batch.
        dropout (float): Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the inputs.
        recurrent_dropout (float): Float between 0 and 1. Fraction of the
            units to drop for the linear transformation of the
            recurrent state.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = ConvGRU2DCell(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             data_format=data_format,
                             dilation_rate=dilation_rate,
                             activation=activation,
                             recurrent_activation=recurrent_activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout,
                             dtype=kwargs.get('dtype'))

        super(ConvGRU2D, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        result = super(ConvGRU2D, self).call(inputs,
                                             mask=mask,
                                             training=training,
                                             initial_state=initial_state)
        return result

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvGRU2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
