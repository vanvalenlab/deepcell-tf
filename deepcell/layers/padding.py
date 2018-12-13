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
from tensorflow.framework import tensor_shape
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        # self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
        # self.input_spec = [InputSpec(ndim=4)]
        # self.data_format = conv_utils.normalize_data_format(data_format)
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        else:
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def call(self, inputs):
        w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            paddings = ((0, 0), (0, 0), h_pad, w_pad)
        else:
            paddings = ((0, 0), h_pad, w_pad, (0, 0))
        return tf.pad(inputs, paddings, 'REFLECT')

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
        super(ReflectionPadding3D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding),
                            (padding, padding),
                            (padding, padding))

        elif hasattr(padding, '__len__'):
            if len(padding) != 3:
                raise ValueError('`padding` should have 3 elements. '
                                 'Found: ' + str(padding))
            dim1_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                      '1st entry of padding')
            dim2_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                      '2nd entry of padding')
            dim3_padding = conv_utils.normalize_tuple(padding[2], 2,
                                                      '3rd entry of padding')
            self.padding = (dim1_padding, dim2_padding, dim3_padding)
        else:
            raise ValueError(
                '`padding` should be either an int, '
                'a tuple of 3 ints '
                '(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad), '
                'or a tuple of 3 tuples of 2 ints '
                '((left_dim1_pad, right_dim1_pad),'
                ' (left_dim2_pad, right_dim2_pad),'
                ' (left_dim3_pad, right_dim2_pad)). '
                'Found: ' + str(padding))

        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                dim1 = input_shape[2] + 2 * self.padding[0][0]
            else:
                dim1 = None
            if input_shape[3] is not None:
                dim2 = input_shape[3] + 2 * self.padding[1][0]
            else:
                dim2 = None
            if input_shape[4] is not None:
                dim3 = input_shape[4] + 2 * self.padding[2][0]
            else:
                dim3 = None
            return tensor_shape.TensorShape(
                [input_shape[0], input_shape[1], dim1, dim2, dim3])
        else:
            if input_shape[1] is not None:
                dim1 = input_shape[1] + 2 * self.padding[0][1]
            else:
                dim1 = None
            if input_shape[2] is not None:
                dim2 = input_shape[2] + 2 * self.padding[1][1]
            else:
                dim2 = None
            if input_shape[3] is not None:
                dim3 = input_shape[3] + 2 * self.padding[2][1]
            else:
                dim3 = None
            return tensor_shape.TensorShape(
                [input_shape[0], dim1, dim2, dim3, input_shape[4]])

    def call(self, x, mask=None):
        z_pad, w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            paddings = ((0, 0), (0, 0), z_pad, h_pad, w_pad)
        else:
            paddings = ((0, 0), z_pad, h_pad, w_pad, (0, 0))
        return tf.pad(x, paddings, 'REFLECT')

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
