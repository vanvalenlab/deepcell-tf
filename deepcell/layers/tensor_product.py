"""
location_layers.py

Layers to generate tensor products for 2D and 3D data

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras._impl.keras.utils import conv_utils


class TensorProd2D(Layer):
    def __init__(self,
                 input_dim,
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
        super(TensorProd2D, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found None')
        input_dim = input_shape[channel_axis]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        backend = K.backend()

        if backend == "theano":
            Exception('This version of DeepCell only works with the tensorflow backend')

        if self.data_format == 'channels_first':
            output = tf.tensordot(inputs, self.kernel, axes=[[1], [0]])
            output = tf.transpose(output, perm=[0, 3, 1, 2])
            # output = K.dot(inputs, self.kernel)

        elif self.data_format == 'channels_last':
            output = tf.tensordot(inputs, self.kernel, axes=[[3], [0]])

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(output)

        return output

    def compute_output_shape(self, input_shape):

        if self.data_format == 'channels_first':
            output_shape = tuple(input_shape[0], self.output_dim, input_shape[2], input_shape[3])
        else:
            output_shape = tuple(input_shape[0], input_shape[1], input_shape[2], self.output_dim)

        return output_shape

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'data_format': self.data_format,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(TensorProd2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TensorProd3D(Layer):
    def __init__(self,
                 input_dim,
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
        super(TensorProd3D, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found None')
        input_dim = input_shape[channel_axis]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(min_ndim=2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        backend = K.backend()

        if backend == "theano":
            Exception('This version of DeepCell only works with the tensorflow backend')

        if self.data_format == 'channels_first':
            output = tf.tensordot(inputs, self.kernel, axes=[[1], [0]])
            output = tf.transpose(output, perm=[0, 4, 1, 2, 3])

        elif self.data_format == 'channels_last':
            output = tf.tensordot(inputs, self.kernel, axes=[[4], [0]])

        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            output_shape = tuple(input_shape[0], self.output_dim, input_shape[2], input_shape[3], input_shape[4])
        else:
            output_shape = tuple(input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)

        return output_shape

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'data_format': self.data_format,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(TensorProd3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
