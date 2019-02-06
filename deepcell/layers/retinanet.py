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
"""RetinaNet layers adapted from https://github.com/fizyr/keras-retinanet"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

try:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils

from deepcell.utils import retinanet_anchor_utils


class Anchors(Layer):
    """Keras layer for generating achors for a given shape.

    Args:
        size: The base size of the anchors to generate.
        stride: The stride of the anchors to generate.
        ratios: The ratios of the anchors to generate,
            defaults to AnchorParameters.default.ratios.
        scales: The scales of the anchors to generate,
            defaults to AnchorParameters.default.scales.
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
    """

    def __init__(self,
                 size,
                 stride,
                 ratios=None,
                 scales=None,
                 data_format=None,
                 *args,
                 **kwargs):
        super(Anchors, self).__init__(*args, **kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios = retinanet_anchor_utils.AnchorParameters.default.ratios
        elif isinstance(ratios, list):
            self.ratios = np.array(ratios)
        if scales is None:
            self.scales = retinanet_anchor_utils.AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales = np.array(scales)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors = K.variable(retinanet_anchor_utils.generate_anchors(
            base_size=size, ratios=ratios, scales=scales))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features_shape = K.shape(inputs)

        # generate proposals from bbox deltas and shifted anchors
        if self.data_format == 'channels_first':
            anchors = retinanet_anchor_utils.shift(
                features_shape[2:4], self.stride, self.anchors)
        else:
            anchors = retinanet_anchor_utils.shift(
                features_shape[1:3], self.stride, self.anchors)
        anchors = tf.tile(K.expand_dims(anchors, axis=0),
                          (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        # input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if None not in input_shape[1:]:
            if self.data_format == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = {
            'size': self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
            'data_format': self.data_format,
        }
        base_config = super(Anchors, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UpsampleLike(Layer):
    """Layer for upsampling a Tensor to be the same shape as another Tensor."""

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
        # input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(UpsampleLike, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RegressBoxes(Layer):
    """Keras layer for applying regression values to boxes."""

    def __init__(self, mean=None, std=None, data_format=None, *args, **kwargs):
        """Initializer for the RegressBoxes layer.

        Args:
            mean: The mean value of the regression values
                which was used for normalization.
            std:  The standard value of the regression values
                which was used for normalization.
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
        """
        super(RegressBoxes, self).__init__(*args, **kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple.'
                             ' Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. '
                             'Received: {}'.format(type(std)))

        self.mean = mean
        self.std = std

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return retinanet_anchor_utils.bbox_transform_inv(
            anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        # input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape[0]

    def get_config(self):
        config = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'data_format': self.data_format
        }
        base_config = super(RegressBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClipBoxes(Layer):
    """Keras layer to clip box values to lie inside a given shape."""

    def __init__(self, data_format=None, **kwargs):
        super(ClipBoxes, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = K.cast(K.shape(image), K.floatx())
        if K.image_data_format() == 'channels_first':
            height = shape[2]
            width = shape[3]
        else:
            height = shape[1]
            width = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height)

        return K.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        # input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape[1]

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(ClipBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
