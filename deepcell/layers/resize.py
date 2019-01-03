# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""Layers to resize input images"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils


class Resize2D(Layer):
    def __init__(self, scale=2, data_format=None, **kwargs):
        super(Resize2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.scale = scale

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]

        rows *= self.scale
        cols *= self.scale

        if self.data_format == 'channels_first':
            output_shape = (input_shape[0], input_shape[1], rows, cols)
        else:
            output_shape = (input_shape[0], rows, cols, input_shape[3])

        return tensor_shape.TensorShape(output_shape)

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
        config = {
            'scale': self.scale,
            'data_format': self.data_format
        }
        base_config = super(Resize2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
