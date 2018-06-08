"""
dc_custom_layers.py

Custom layers for convolutional neural networks

@author: David Van Valen
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras._impl.keras.utils import conv_utils

"""
Custom layers
"""

class ImageNormalization2D(Layer):
    def __init(self, norm_method=None, filter_size=61, **kwargs):
        super(ImageNormalization2D, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.norm_method = norm_method
        self.data_format = kwargs.get('data_format', K.image_data_format())

        if self.data_format == 'channels_first':
            self.channel_axis = 1
        else:
            self.channel_axis = -1

    def compute_output_shape(self, input_shape):
        return input_shape

    def _window_std_filter(self, inputs, epsilon=K.epsilon()):
        c1 = self._average_filter(inputs)
        c2 = self._average_filter(tf.square(inputs))
        output = tf.sqrt(c2 - c1 * c1) + epsilon
        return output

    def _average_filter(self, inputs):
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            in_channels = input_shape[1]
        else:
            in_channels = input_shape[-1]

        W = np.ones((self.filter_size, self.filter_size, in_channels, 1))
        W /= W.size
        kernel = tf.Variable(W.astype(K.floatx()))

        outputs = tf.nn.depthwise_conv2d(inputs, kernel, [1, 1, 1, 1], 'SAME')

        return outputs

    def _reduce_median(self, inputs, axes=None):
        rank = tf.rank(inputs)
        reduce_axes = axes
        axes_to_keep = [axis for axis in range(rank) if axis not in reduce_axes]
        input_shape = tf.shape(inputs)

        new_shape = [input_shape[axis] for axis in axes_to_keep]
        new_shape.append(-1)

        reshaped_inputs = tf.reshape(inputs, new_shape)

        median_index = reshaped_inputs.get_shape()[-1] // 2

        median = tf.nn.top_k(reshaped_inputs, k=median_index)

        return median

    def call(self, inputs):

        if self.norm_method == 'std':
            outputs = inputs - self._average_filter(inputs)
            outputs /= self._window_std(outputs)

        elif self.norm_method == 'max':
            outputs = inputs / tf.reduce_max(inputs)
            outputs -= self._average_filter(outputs)

        else:
            reduce_axes = [1, 2, 3]
            reduce_axes.remove(self.channel_axis)
            mean = tf.reduce_median(inputs, axes=reduce_axes)
            outputs = inputs / mean
            outputs -= self._average_filter(outputs)

        return outputs

    def get_config(self):
        config = {'process_std': self.process_std,
                  'data_format': self.data_format}
        base_config = super(ImageNormalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Location(Layer):
    def __init__(self, in_shape, data_format=None, **kwargs):
        super(Location, self).__init__(**kwargs)
        self.in_shape = in_shape
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], 2, input_shape[2], input_shape[3])

        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1], input_shape[2], 2)

    def call(self, inputs):
        input_shape = self.in_shape

        if self.data_format == 'channels_last':
            x = tf.range(0, input_shape[0], dtype=K.floatx())
            y = tf.range(0, input_shape[1], dtype=K.floatx())

        else:
            x = tf.range(0, input_shape[1], dtype=K.floatx())
            y = tf.range(0, input_shape[2], dtype=K.floatx())

        x = tf.divide(x, tf.reduce_max(x))
        y = tf.divide(y, tf.reduce_max(y))

        loc_x, loc_y = tf.meshgrid(y, x)

        if self.data_format == 'channels_last':
            loc = tf.stack([loc_x, loc_y], axis=-1)
        else:
            loc = tf.stack([loc_x, loc_y], axis=0)


        location = tf.expand_dims(loc, 0)

        return location

    def get_config(self):
        config = {'in_shape': self.in_shape,
                  'data_format': self.data_format}
        base_config = super(Location, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Location3D(Layer):
    def __init__(self, in_shape, data_format=None, **kwargs):
        super(Location3D, self).__init__(**kwargs)
        self.in_shape = in_shape
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0], 2, input_shape[2], input_shape[3], input_shape[4])

        elif self.data_format == 'channels_last':
            return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 2)

    def call(self, inputs):
        input_shape = self.in_shape

        if self.data_format == 'channels_last':
            x = tf.range(0, input_shape[2], dtype=K.floatx())
            y = tf.range(0, input_shape[3], dtype=K.floatx())

        else:
            x = tf.range(0, input_shape[3], dtype=K.floatx())
            y = tf.range(0, input_shape[4], dtype=K.floatx())

        x = tf.divide(x, tf.reduce_max(x))
        y = tf.divide(y, tf.reduce_max(y))

        loc_x, loc_y = tf.meshgrid(y, x)

        if self.data_format == 'channels_last':
            loc = tf.stack([loc_x, loc_y], axis=-1)
        else:
            loc = tf.stack([loc_x, loc_y], axis=0)

        if self.data_format == 'channels_last':
            location = tf.expand_dims(loc, 0)
        else:
            location = tf.expand_dims(loc, 1)

        number_of_frames = input_shape[1] if self.data_format == 'channels_last' else input_shape[2]

        location_list = [tf.identity(location) for _ in range(number_of_frames)]

        if self.data_format == 'channels_last':
            location_concat = tf.concat(location_list, axis=0)
        else:
            location_concat = tf.concat(location_list, axis=1)

        location_output = tf.expand_dims(location_concat, 0)

        return location_output

    def get_config(self):
        config = {'in_shape': self.in_shape,
                  'data_format': self.data_format}
        base_config = super(Location3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

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
        config = {'scale': self.scale,
                  'data_format': self.data_format}
        base_config = super(Resize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class dilated_MaxPool2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, dilation_rate=1, padding='valid',
                 data_format=None, **kwargs):
        super(dilated_MaxPool2D, self).__init__(**kwargs)
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
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.pool_size[0], padding=self.padding,
                                             stride=self.strides[0], dilation=self.dilation_rate)

        cols = conv_utils.conv_output_length(cols, self.pool_size[1], padding=self.padding,
                                             stride=self.strides[1], dilation=self.dilation_rate)

        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def  _pooling_function(self, inputs, pool_size, dilation_rate, strides, padding, data_format):
        backend = K.backend()

        #dilated pooling for tensorflow backend
        if backend == "theano":
            Exception('This version of DeepCell only works with the tensorflow backend')

        if data_format == 'channels_first':
            df = 'NCHW'
        elif data_format == 'channels_last':
            df = 'NHWC'

        if self.padding == "valid":
            padding_input = "VALID"

        if self.padding == "same":
            padding_input = "SAME"

        output = tf.nn.pool(inputs, window_shape=pool_size, pooling_type="MAX",
                            padding=padding_input, dilation_rate=(dilation_rate, dilation_rate),
                            strides=strides, data_format=df)

        return output

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        dilation_rate=self.dilation_rate,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'dilation_rate': self.dilation_rate,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(dilated_MaxPool2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

        elif self.data_format == 'channels_last':
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

        elif self.data_format == 'channels_last':
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
