# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils


class UpsampleLike(Layer):
    """Layer for upsampling a Tensor to be the same shape as another Tensor.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """

    def __init__(self, data_format=None, **kwargs):
        super(UpsampleLike, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def _resize_drop_axis(self, image, size, axis):
        image_shape = tf.shape(image)

        new_shape = []
        axes_resized = list(set([0, 1, 2, 3, 4]) - set([0, 4, axis]))
        for ax in range(K.ndim(image) - 1):
            if ax != axis:
                new_shape.append(image_shape[ax])
            if ax == 3:
                new_shape.append(image_shape[-1] * image_shape[axis])

        new_shape_2 = []
        for ax in range(K.ndim(image)):
            if ax == 0 or ax == 4 or ax == axis:
                new_shape_2.append(image_shape[ax])
            elif ax == axes_resized[0]:
                new_shape_2.append(size[0])
            elif ax == axes_resized[1]:
                new_shape_2.append(size[1])

        new_image = tf.reshape(image, new_shape)
        new_image_resized = tf.image.resize(
            new_image,
            size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        new_image_2 = tf.reshape(new_image_resized, new_shape_2)

        return new_image_2

    def resize_volumes(self, volume, size):
        # TODO: K.resize_volumes?
        if self.data_format == 'channels_first':
            volume = tf.transpose(volume, (0, 2, 3, 4, 1))
            new_size = (size[2], size[3], size[4])
        else:
            new_size = (size[1], size[2], size[3])

        new_shape_0 = (new_size[1], new_size[2])
        new_shape_1 = (new_size[0], new_size[1])

        resized_volume = self._resize_drop_axis(volume, new_shape_0, axis=1)
        resized_volume = self._resize_drop_axis(resized_volume, new_shape_1, axis=3)

        new_shape_static = [None, None, None, None, volume.get_shape()[-1]]
        resized_volume.set_shape(new_shape_static)

        if self.data_format == 'channels_first':
            resized_volume = tf.transpose(resized_volume, (0, 4, 1, 2, 3))

        return resized_volume

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        if source.get_shape().ndims == 4:
            if self.data_format == 'channels_first':
                source = tf.transpose(source, (0, 2, 3, 1))
                new_shape = (target_shape[2], target_shape[3])
                # TODO: K.resize_images?
                output = tf.image.resize(
                    source, new_shape,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                output = tf.transpose(output, (0, 3, 1, 2))
                return output
            new_shape = (target_shape[1], target_shape[2])
            return tf.image.resize(
                source, new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if source.get_shape().ndims == 5:
            output = self.resize_volumes(source, target_shape)
            return output

        else:
            raise ValueError('Expected input[0] to have ndim of 4 or 5, found'
                             ' %s.' % source.get_shape().ndims)

    def compute_output_shape(self, input_shape):
        in_0 = tensor_shape.TensorShape(input_shape[0]).as_list()
        in_1 = tensor_shape.TensorShape(input_shape[1]).as_list()
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(([in_0[0], in_0[1]] + in_1[2:]))
        return tensor_shape.TensorShape(([in_0[0]] + in_1[1:-1] + [in_0[-1]]))

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(UpsampleLike, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Upsample(Layer):
    """Upsample layer adapted from https://github.com/fizyr/keras-maskrcnn.

    Args:
        target_size (tuple): 2D tuple for target size ``(x, y)``.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """

    def __init__(self, target_size, data_format=None, *args, **kwargs):
        self.target_size = target_size
        self.data_format = conv_utils.normalize_data_format(data_format)
        super(Upsample, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        new_shape = (self.target_size[0], self.target_size[1])
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, (0, 2, 3, 1))
        outputs = tf.image.resize(
            inputs, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, (0, 3, 1, 2))
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            output_shape = (
                input_shape[0],
                input_shape[1],
                self.target_size[0],
                self.target_size[1])
        else:
            output_shape = (
                input_shape[0],
                self.target_size[0],
                self.target_size[1],
                input_shape[-1])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            'target_size': self.target_size,
            'data_format': self.data_format
        }
        base_config = super(Upsample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
