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
"""Layers for padding for 2D and 3D images
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.layers import InputSpec
try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
        self.input_spec = [InputSpec(ndim=4)]
        self.data_format = conv_utils.normalize_data_format(data_format)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            output_shape = (input_shape[0],
                            input_shape[1],
                            input_shape[2] + 2 * self.padding[0],
                            input_shape[3] + 2 * self.padding[1])
        else:
            output_shape = (input_shape[0],
                            input_shape[1] + 2 * self.padding[0],
                            input_shape[2] + 2 * self.padding[1],
                            input_shape[3])

        return tensor_shape.TensorShape(output_shape)

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            paddings = [[0, 0], [0, 0], [h_pad, h_pad], [w_pad, w_pad]]
        else:
            paddings = [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]]
        return tf.pad(x, paddings, 'REFLECT')


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
        self.padding = conv_utils.normalize_tuple(padding, 3, 'padding')
        self.input_spec = [InputSpec(ndim=4)]
        self.data_format = conv_utils.normalize_data_format(data_format)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            output_shape = (input_shape[0],
                            input_shape[1],
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
            paddings = [[0, 0], [0, 0], [z_pad, z_pad], [h_pad, h_pad], [w_pad, w_pad]]
        else:
            paddings = [[0, 0], [z_pad, z_pad], [h_pad, h_pad], [w_pad, w_pad], [0, 0]]
        return tf.pad(x, paddings, 'REFLECT')
