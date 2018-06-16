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
            output_shape = (input_shape[0], 2, input_shape[2], input_shape[3])
        else:
            output_shape = (input_shape[0], input_shape[1], input_shape[2], 2)
        return output_shape

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
        config = {
            'in_shape': self.in_shape,
            'data_format': self.data_format
        }
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
            output_shape = (input_shape[0], 2, input_shape[2], input_shape[3], input_shape[4])
        else:
            output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 2)
        return output_shape

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
        config = {
            'in_shape': self.in_shape,
            'data_format': self.data_format
        }
        base_config = super(Location3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
