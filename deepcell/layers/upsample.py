# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/tf-keras-retinanet/LICENSE
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
"""Upsampling layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils


class UpsampleLike(Layer):
    """Layer for upsampling a Tensor to be the same shape as another Tensor.

    Adapted from https://github.com/fizyr/keras-retinanet.
    """

    def __init__(self, data_format=None, **kwargs):
        super(UpsampleLike, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        if self.data_format == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            new_shape = (target_shape[2], target_shape[3])
            output = tf.image.resize_images(
                source, new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.transpose(output, (0, 3, 1, 2))
            return output
        new_shape = (target_shape[1], target_shape[2])
        return tf.image.resize_images(
            source, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        in_0 = tensor_shape.TensorShape(input_shape[0]).as_list()
        in_1 = tensor_shape.TensorShape(input_shape[1]).as_list()
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(([in_0[0], in_0[1]] + in_1[2:4]))
        return tensor_shape.TensorShape(([in_0[0]] + in_1[1:3] + [in_0[-1]]))

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(UpsampleLike, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Upsample(Layer):
    """Upsample layer adapted from https://github.com/fizyr/keras-maskrcnn."""

    def __init__(self, target_size, data_format=None, *args, **kwargs):
        self.target_size = target_size
        self.data_format = conv_utils.normalize_data_format(data_format)
        super(Upsample, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        new_shape = (self.target_size[0], self.target_size[1])
        return tf.resize_images(
            inputs, new_shape,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        new_shape = tuple([input_shape[0], self.target_size, input_shape[-1]])
        return tensor_shape.TensorShape(new_shape)

    def get_config(self):
        config = {
            'target_size': self.target_size,
            'data_format': self.data_format
        }
        base_config = super(Upsample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
