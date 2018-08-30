"""
padding.py
Layers for padding for 2D and 3D images
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import InputSpec
try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], input_shape[2] + 2 * self.padding[0], input_shape[3] + 2 * self.padding[1])
        else:
            output_shape = (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])
        
        return tensor_shape.TensorShape(output_shape)

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            return tf.pad(x, [[0,0], [0,0], [h_pad,h_pad], [w_pad,w_pad]], 'REFLECT')
        else:
            return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0]], 'REFLECT')

class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1],
                input_shape[2] + 2 * self.padding[0], 
                input_shape[3] + 2 * self.padding[1], 
                input_shape[4] + 2 * self.padding[2])
        else:
            output_shape = (input_shape[0], 
                input_shape[1] + 2 * self.padding[0], 
                input_shape[2] + 2 * self.padding[1], 
                input_shape[3] + 2 * self.padding[2], 
                input_shape[4])
        
        return tensor_shape.TensorShape(output_shape)

    def call(self, x, mask=None):
        z_pad, w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            return tf.pad(x, [[0,0], [0,0], [z_pad, z_pad], [h_pad,h_pad], [w_pad,w_pad]], 'REFLECT')
        else:
            return tf.pad(x, [[0,0], [z_pad, z_pad], [h_pad,h_pad], [w_pad,w_pad], [0,0]], 'REFLECT')