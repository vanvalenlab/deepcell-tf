# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Layers to encode location data
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
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
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], 2, input_shape[2], input_shape[3])
        else:
            output_shape = (input_shape[0], input_shape[1], input_shape[2], 2)
        return tensor_shape.TensorShape(output_shape)

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
        location = tf.tile(location, [tf.shape(inputs)[0], 1, 1, 1])

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
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], 3, input_shape[2], input_shape[3], input_shape[4])
        else:
            output_shape = (input_shape[0], input_shape[1], input_shape[2], input_shape[3], 3)
        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        input_shape = self.in_shape

        if self.data_format == 'channels_last':
            z = tf.range(0, input_shape[0], dtype=K.floatx())
            x = tf.range(0, input_shape[1], dtype=K.floatx())
            y = tf.range(0, input_shape[2], dtype=K.floatx())
        else:
            z = tf.range(0, input_shape[1], dtype=K.floatx())
            x = tf.range(0, input_shape[2], dtype=K.floatx())
            y = tf.range(0, input_shape[3], dtype=K.floatx())

        x = tf.divide(x, tf.reduce_max(x))
        y = tf.divide(y, tf.reduce_max(y))
        z = tf.divide(z, tf.reduce_max(z))

        loc_z, loc_x, loc_y = tf.meshgrid(z, x, y, indexing='ij')

        if self.data_format == 'channels_last':
            loc = tf.stack([loc_z, loc_x, loc_y], axis=-1)
        else:
            loc = tf.stack([loc_z, loc_x, loc_y], axis=0)

        location = tf.expand_dims(loc, 0)
        location = tf.tile(location, [tf.shape(inputs)[0], 1, 1, 1, 1])

        return location

    def get_config(self):
        config = {
            'in_shape': self.in_shape,
            'data_format': self.data_format
        }
        base_config = super(Location3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
