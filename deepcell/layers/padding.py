# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""Layers for padding for 2D and 3D images
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import ZeroPadding3D


class ReflectionPadding2D(ZeroPadding2D):
    """Reflection-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns of reflected values
    at the top, bottom, left and right side of an image tensor.

    Arguments:
        padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to height and width.
            - If tuple of 2 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_height_pad, symmetric_width_pad)`.
            - If tuple of 2 tuples of 2 ints:
                interpreted as
                `((top_pad, bottom_pad), (left_pad, right_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    Input shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    Output shape:
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, padded_rows, padded_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, padded_rows, padded_cols)`
    """

    def call(self, inputs):
        w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], list(w_pad), list(h_pad)]
        else:
            pattern = [[0, 0], list(w_pad), list(h_pad), [0, 0]]
        return array_ops.pad(inputs, pattern, mode='REFLECT')


class ReflectionPadding3D(ZeroPadding3D):
    """Reflection-padding layer for 3D data (spatial or spatio-temporal).
    Arguments:
        padding: int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
            - If int: the same symmetric padding
                is applied to height and width.
            - If tuple of 3 ints:
                interpreted as two different
                symmetric padding values for height and width:
                `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
            - If tuple of 3 tuples of 2 ints:
                interpreted as
                `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
                    right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
            while `channels_first` corresponds to inputs with shape
            `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    Input shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
                depth)`
        - If `data_format` is `"channels_first"`:
            `(batch, depth, first_axis_to_pad, second_axis_to_pad,
                third_axis_to_pad)`
    Output shape:
        5D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad,
                depth)`
        - If `data_format` is `"channels_first"`:
            `(batch, depth, first_padded_axis, second_padded_axis,
                third_axis_to_pad)`
    """

    def call(self, inputs):
        d_pad, w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], [d_pad[0], d_pad[1]],
                       [w_pad[0], w_pad[1]], [h_pad[0], h_pad[1]]]
        else:
            pattern = [[0, 0], [d_pad[0], d_pad[1]],
                       [w_pad[0], w_pad[1]], [h_pad[0], h_pad[1]], [0, 0]]
        return array_ops.pad(inputs, pattern, mode='REFLECT')
