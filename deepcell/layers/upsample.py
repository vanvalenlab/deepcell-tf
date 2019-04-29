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
        return tf.image.resize_images(
            inputs, new_shape,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        output_shape = tuple([
            input_shape[0],
            self.target_size[0],
            self.target_size[1],
            input_shape[-1]])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            'target_size': self.target_size,
            'data_format': self.data_format
        }
        base_config = super(Upsample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Upsample3D(Layer):
    """Upsample layer adapted from https://github.com/fizyr/keras-maskrcnn."""

    def __init__(self, target_size, data_format=None, *args, **kwargs):
        self.target_size = target_size
        self.data_format = conv_utils.normalize_data_format(data_format)
        super(Upsample3D, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return tf.keras.backend.resize_volumes(
            inputs, self.target_size[0], self.target_size[1], self.target_size[2],
            self.data_format)

    def compute_output_shape(self, input_shape):
        output_shape = tuple([
            input_shape[0],
            self.target_size[0],
            self.target_size[1],
            self.target_size[2],
            input_shape[-1]])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            'target_size': self.target_size,
            'data_format': self.data_format
        }
        base_config = super(Upsample3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class UpsampleLike3D(Layer):
    """Layer for upsampling a Tensor to be the same shape as another Tensor.

    Adapted from https://github.com/fizyr/keras-retinanet.
    """

    def __init__(self, data_format=None, **kwargs):
        super(UpsampleLike3D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)



        if self.data_format == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            output = tf.keras.backend.resize_volumes(
                source, target_shape[2], target_shape[3],target_shape[4],
                self.data_format)

            output = tf.transpose(output, (0, 3, 1, 2))
            return output

        # Reshape source as a 4D tensor
        reshaped = tf.reshape(source, [tf.shape(source)[0],  source.get_shape()[1], source.get_shape()[2], source.get_shape()[3] * source.get_shape()[4]])

        # Resize the first two dimensions of the tensor
        new_size = tf.constant([int(target.get_shape()[1]), int(target.get_shape()[2])])
        resized = tf.image.resize_images(reshaped, new_size)

        # Undo the reshape
        undo_reshape = tf.reshape(resized,  [tf.shape(source)[0],  target.get_shape()[1], target.get_shape()[2], source.get_shape()[3], source.get_shape()[4]])

        # Switch the two first dimension with the two last one
        transposed = tf.transpose(undo_reshape, [0, 3, 4, 1, 2])

        # Reshape as a 4D tensor
        reshaped2 = tf.reshape(transposed, [tf.shape(transposed)[0],  transposed.get_shape()[1], transposed.get_shape()[2], transposed.get_shape()[3] * transposed.get_shape()[4]])

        # Resize the new two first dimension
        new_size2 = tf.constant([int(target.get_shape()[3]), int(source.get_shape()[4])])
        resized2 = tf.image.resize_images(reshaped2, new_size2)

        # Undo reshape
        undo_reshape2 = tf.reshape(resized2,  [tf.shape(source)[0],  target.get_shape()[3], source.get_shape()[4], target.get_shape()[1], target.get_shape()[2]])

        #Undo transpose
        undo_transpose = tf.transpose(undo_reshape2, [0,3,4,1,2] )

        return undo_transpose

    def compute_output_shape(self, input_shape):
        in_0 = tensor_shape.TensorShape(input_shape[0]).as_list()
        in_1 = tensor_shape.TensorShape(input_shape[1]).as_list()
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(([in_0[0], in_0[1]] + in_1[2:4]))
        return tensor_shape.TensorShape(([in_0[0]] + in_1[1:3] + [in_0[-1]]))

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(UpsampleLike3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

