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
"""RetinaNet layers adapted from https://github.com/fizyr/keras-retinanet"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils

from deepcell.utils import retinanet_anchor_utils


class Anchors(Layer):
    """Keras layer for generating achors for a given shape.

    Args:
        size: The base size of the anchors to generate.
        stride: The stride of the anchors to generate.
        ratios: The ratios of the anchors to generate,
            defaults to ``AnchorParameters.default.ratios``.
        scales: The scales of the anchors to generate,
            defaults to ``AnchorParameters.default.scales``.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
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

        if self.ratios is None:
            self.ratios = retinanet_anchor_utils.AnchorParameters.default.ratios
        if isinstance(self.ratios, list):
            self.ratios = np.array(self.ratios)
        if self.scales is None:
            self.scales = retinanet_anchor_utils.AnchorParameters.default.scales
        if isinstance(self.scales, list):
            self.scales = np.array(self.scales)

        self.num_anchors = len(self.ratios) * len(self.scales)
        self.anchors = K.variable(retinanet_anchor_utils.generate_anchors(
            base_size=size, ratios=ratios, scales=scales))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features_shape = K.shape(inputs)

        # generate proposals from bbox deltas and shifted anchors
        row_axis = 2 if self.data_format == "channels_first" else 1
        anchors = retinanet_anchor_utils.shift(
            features_shape[row_axis:row_axis + 2], self.stride, self.anchors)

        anchors = tf.tile(K.expand_dims(anchors, axis=0),
                          (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if None not in input_shape[1:]:
            if self.data_format == "channels_first":
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return tensor_shape.TensorShape((input_shape[0], total, 4))
        return tensor_shape.TensorShape((input_shape[0], None, 4))

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


class RegressBoxes(Layer):
    """Layer for applying regression values to boxes.

    Args:
        mean: The mean value of the regression values
            which was used for normalization.
        std:  The standard value of the regression values
            which was used for normalization.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """

    def __init__(self, mean=None, std=None, data_format=None, *args, **kwargs):
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
        return tensor_shape.TensorShape(input_shape[0])

    def get_config(self):
        config = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'data_format': self.data_format
        }
        base_config = super(RegressBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClipBoxes(Layer):
    """Keras layer to clip box values to lie inside a given shape.

    Args:
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """

    def __init__(self, data_format=None, **kwargs):
        super(ClipBoxes, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = K.cast(K.shape(image), K.floatx())
        ndim = K.ndim(image)
        if self.data_format == "channels_first":
            height = shape[ndim - 2]
            width = shape[ndim - 1]
        else:
            height = shape[ndim - 3]
            width = shape[ndim - 2]

        x1, y1, x2, y2 = tf.unstack(boxes, axis=-1)
        x1 = tf.clip_by_value(x1, 0, width - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)
        x2 = tf.clip_by_value(x2, 0, width - 1)
        y2 = tf.clip_by_value(y2, 0, height - 1)
        return K.stack([x1, y1, x2, y2], axis=ndim - 2)

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(input_shape[1]).as_list()

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(ClipBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConcatenateBoxes(Layer):
    """Keras layer to concatenate bouding boxes."""
    def call(self, inputs, **kwargs):
        boxes, other = inputs
        boxes_shape = K.shape(boxes)
        n = int(K.ndim(boxes) - 1)
        other_shape = tuple([boxes_shape[i] for i in range(n)] + [-1])
        other = K.reshape(other, other_shape)
        return K.concatenate([boxes, other], axis=K.ndim(boxes) - 1)

    def compute_output_shape(self, input_shape):
        boxes_shape, other_shape = input_shape
        n = len(boxes_shape) - 1
        output_shape = tuple(list(boxes_shape[:n]) +
                             [K.prod([s for s in other_shape[n:]]) + 4])
        return tensor_shape.TensorShape(output_shape)


class _RoiAlign(Layer):
    """Original RoiAlign Layer from https://github.com/fizyr/keras-retinanet

    Args:
        crop_size (tuple): 2-length tuple of integers,
            the ROIs get cropped to this size.
        parallel_iterations (int): Number of parallel mappings to use
            for ROI.
    """
    def __init__(self, crop_size=(14, 14), parallel_iterations=32, **kwargs):
        self.crop_size = crop_size
        self.parallel_iterations = parallel_iterations
        super(_RoiAlign, self).__init__(**kwargs)

    def map_to_level(self, boxes,
                     canonical_size=224,
                     canonical_level=1,
                     min_level=0,
                     max_level=4):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        w = x2 - x1
        h = y2 - y1

        size = K.sqrt(w * h)

        log = K.log(size / canonical_size + K.epsilon())
        log2 = log / K.log(K.cast(2, log.dtype))

        levels = tf.floor(canonical_level + log2)
        levels = K.clip(levels, min_level, max_level)

        return levels

    def call(self, inputs, **kwargs):
        image_shape = K.cast(inputs[0], K.floatx())
        boxes = K.stop_gradient(inputs[1])
        scores = K.stop_gradient(inputs[2])
        fpn = [K.stop_gradient(i) for i in inputs[3:]]

        time_distributed = K.ndim(boxes) == 4

        if time_distributed:
            image_shape = image_shape[1:]

            boxes_shape = tf.shape(boxes)
            scores_shape = tf.shape(scores)
            fpn_shape = [tf.shape(f) for f in fpn]

            new_boxes_shape = [-1] + [boxes_shape[i] for i in range(2, K.ndim(boxes))]
            new_scores_shape = [-1] + [scores_shape[i] for i in range(2, K.ndim(scores))]
            new_fpn_shape = [[-1] + [f_s[i] for i in range(2, K.ndim(f))]
                             for f, f_s in zip(fpn, fpn_shape)]

            boxes = tf.reshape(boxes, new_boxes_shape)
            scores = tf.reshape(scores, new_scores_shape)
            fpn = [tf.reshape(f, f_s) for f, f_s in zip(fpn, new_fpn_shape)]

        def _roi_align(args):
            boxes = args[0]
            scores = args[1]
            fpn = args[2]

            # compute from which level to get features from
            target_levels = self.map_to_level(boxes)

            # process each pyramid independently
            rois, ordered_indices = [], []
            for i in range(len(fpn)):
                # select the boxes and classification from this pyramid level
                indices = tf.where(K.equal(target_levels, i))
                ordered_indices.append(indices)

                level_boxes = tf.gather_nd(boxes, indices)
                fpn_shape = K.cast(K.shape(fpn[i]), dtype=K.floatx())

                # convert to expected format for crop_and_resize
                x1 = level_boxes[:, 0]
                y1 = level_boxes[:, 1]
                x2 = level_boxes[:, 2]
                y2 = level_boxes[:, 3]
                level_boxes = K.stack([
                    (y1 / image_shape[1] * fpn_shape[0]) / (fpn_shape[0] - 1),
                    (x1 / image_shape[2] * fpn_shape[1]) / (fpn_shape[1] - 1),
                    (y2 / image_shape[1] * fpn_shape[0] - 1) / (fpn_shape[0] - 1),
                    (x2 / image_shape[2] * fpn_shape[1] - 1) / (fpn_shape[1] - 1),
                ], axis=1)

                # append the rois to the list of rois
                rois.append(tf.image.crop_and_resize(
                    K.expand_dims(fpn[i], axis=0),
                    level_boxes,
                    tf.zeros((K.shape(level_boxes)[0],), dtype='int32'),
                    self.crop_size
                ))

            # concatenate rois to one blob
            rois = K.concatenate(rois, axis=0)

            # reorder rois back to original order
            indices = K.concatenate(ordered_indices, axis=0)
            rois = tf.scatter_nd(indices, rois, K.cast(K.shape(rois), 'int64'))

            return rois

        roi_batch = tf.map_fn(
            _roi_align,
            elems=[boxes, scores, fpn],
            dtype=K.floatx(),
            parallel_iterations=self.parallel_iterations
        )

        if time_distributed:
            roi_shape = tf.shape(roi_batch)
            new_roi_shape = [boxes_shape[0], boxes_shape[1]] + \
                            [roi_shape[i] for i in range(1, K.ndim(roi_batch))]
            roi_batch = tf.reshape(roi_batch, new_roi_shape)

        return roi_batch

    def compute_output_shape(self, input_shape):
        if len(input_shape[3]) == 4:
            output_shape = [
                input_shape[1][0],
                None,
                self.crop_size[0],
                self.crop_size[1],
                input_shape[3][-1]
            ]
            return tensor_shape.TensorShape(output_shape)
        elif len(input_shape[3]) == 5:
            output_shape = [
                input_shape[1][0],
                input_shape[3][1],
                None,
                self.crop_size[0],
                self.crop_size[1],
                input_shape[3][-1]
            ]
            return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {'crop_size': self.crop_size}
        base_config = super(_RoiAlign, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RoiAlign(_RoiAlign):
    """Modified RoiAlign layer.

    Only takes in one feature map, which must be the size of the original image
    """

    def call(self, inputs, **kwargs):
        boxes = K.stop_gradient(inputs[0])
        fpn = K.stop_gradient(inputs[1])

        time_distributed = K.ndim(boxes) == 4

        if time_distributed:
            boxes_shape = K.shape(boxes)
            fpn_shape = K.shape(fpn)

            new_boxes_shape = [-1] + [boxes_shape[i] for i in range(2, K.ndim(boxes))]
            new_fpn_shape = [-1] + [fpn_shape[i] for i in range(2, K.ndim(fpn))]

            boxes = K.reshape(boxes, new_boxes_shape)
            fpn = K.reshape(fpn, new_fpn_shape)

        image_shape = K.cast(K.shape(fpn), K.floatx())

        def _roi_align(args):
            boxes = args[0]
            fpn = args[1]            # process the feature map
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            fpn_shape = K.cast(K.shape(fpn), dtype=K.floatx())
            norm_boxes = K.stack([
                (y1 / image_shape[1] * fpn_shape[0]) / (fpn_shape[0] - 1),
                (x1 / image_shape[2] * fpn_shape[1]) / (fpn_shape[1] - 1),
                (y2 / image_shape[1] * fpn_shape[0] - 1) / (fpn_shape[0] - 1),
                (x2 / image_shape[2] * fpn_shape[1] - 1) / (fpn_shape[1] - 1)
            ], axis=1)

            rois = tf.image.crop_and_resize(
                K.expand_dims(fpn, axis=0),
                norm_boxes,
                tf.zeros((K.shape(norm_boxes)[0],), dtype='int32'),
                self.crop_size)

            return rois

        roi_batch = tf.map_fn(
            _roi_align,
            elems=[boxes, fpn],
            dtype=K.floatx(),
            parallel_iterations=self.parallel_iterations)

        if time_distributed:
            roi_shape = tf.shape(roi_batch)
            new_roi_shape = [boxes_shape[0], boxes_shape[1]] + \
                            [roi_shape[i] for i in range(1, K.ndim(roi_batch))]
            roi_batch = tf.reshape(roi_batch, new_roi_shape)

        return roi_batch


class Shape(Layer):
    """Layer that returns the shape of the input tensor"""
    def call(self, inputs, **kwargs):
        return K.shape(inputs)

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape((len(input_shape),))


class Cast(Layer):
    """Layer that casts the the input to another dtype.

    Args:
        dtype (str): String or ``tf.dtype``, the desired dtype.
    """
    def __init__(self, dtype=None, *args, **kwargs):
        if dtype is None:
            dtype = K.floatx()
        self.dtype = dtype
        super(Cast, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        outputs = K.cast(inputs, self.dtype)
        return outputs
