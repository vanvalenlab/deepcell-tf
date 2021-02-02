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
"""Filter Detection Layer adapted from https://github.com/fizyr/keras-retinanet"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils


def filter_detections(boxes,
                      classification,
                      other=[],
                      class_specific_filter=True,
                      nms=True,
                      score_threshold=0.05,
                      max_detections=300,
                      nms_threshold=0.5):
    """Filter detections using the boxes and classification values.

    Args:
        boxes (numpy.array): Tensor of shape ``(num_boxes, 4)`` containing the
            boxes in ``(x1, y1, x2, y2)`` format.
        classification (numpy.array): Tensor of shape
            ``(num_boxes, num_classes)`` containing the classification scores.
        other (list): List of tensors of shape ``(num_boxes, ...)`` to filter
            along with the boxes and classification scores.
        class_specific_filter (bool): Whether to perform filtering per class,
            or take the best scoring class and filter those.
        nms (bool): Whether to enable non maximum suppression.
        score_threshold (float): Threshold used to prefilter the boxes with.
        max_detections (int): Maximum number of detections to keep.
        nms_threshold (float): Threshold for the IoU value to determine when a
            box should be suppressed.

    Returns:
        list: A list of [``boxes, scores, labels, other[0], other[1], ...]``.
        ``boxes`` is shaped ``(max_detections, 4)`` and contains the
        ``(x1, y1, x2, y2)`` of the non-suppressed boxes.
        ``scores`` is shaped ``(max_detections,)`` and contains the scores
        of the predicted class.
        ``labels`` is shaped ``(max_detections,)`` and contains the
        predicted label.
        ``other[i]`` is shaped ``(max_detections, ...)`` and contains the
        filtered ``other[i]`` data.
        In case there are less than ``max_detections`` detections,
        the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = tf.where(K.greater(scores, score_threshold))

        if nms:
            filtered_boxes = tf.gather_nd(boxes, indices)
            filtered_scores = K.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(
                filtered_boxes,
                filtered_scores,
                max_output_size=max_detections,
                iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = K.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = K.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(K.int_shape(classification)[1]):
            scores = classification[:, c]
            labels = c * tf.ones((K.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = K.concatenate(all_indices, axis=0)
    else:
        scores = K.max(classification, axis=1)
        labels = K.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(
        scores, k=K.minimum(max_detections, K.shape(scores)[0]))

    # filter input using the final set of indices
    indices = K.gather(indices[:, 0], top_indices)
    boxes = K.gather(boxes, indices)
    labels = K.gather(labels, top_indices)
    other_ = [K.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = K.maximum(0, max_detections - K.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = K.cast(labels, 'int32')
    pads = lambda x: [[0, pad_size]] + [[0, 0] for _ in range(1, K.ndim(x))]
    other_ = [tf.pad(o, pads(o), constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(K.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_


class FilterDetections(Layer):
    """Keras layer for filtering detections using score threshold and NMS.

    Args:
        nms (bool): Whether to enable non maximum suppression.
        class_specific_filter (bool): Whether to perform filtering per class,
            or take the best scoring class and filter those.
        nms_threshold (float): Threshold for the IoU value to determine when a
            box should be suppressed.
        score_threshold (float): Threshold used to prefilter the boxes with.
        max_detections (int): Maximum number of detections to keep.
        parallel_iterations (int): Number of batch items to process in parallel.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.

    """
    def __init__(self,
                 nms=True,
                 class_specific_filter=False,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 max_detections=300,
                 parallel_iterations=32,
                 data_format=None,
                 **kwargs):
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.data_format = conv_utils.normalize_data_format(data_format)
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """Constructs the NMS graph.

        Args:
            inputs: List of
                ``[boxes, classification, other[0], other[1], ...]`` tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        other = inputs[2:]

        time_distributed = K.ndim(boxes) == 4

        if time_distributed:
            boxes_shape = K.shape(boxes)
            # classification_shape = classification.get_shape()
            classification_shape = K.shape(classification)
            other_shape = [K.shape(o) for o in other]

            new_boxes_shape = [-1] + [boxes_shape[i] for i in range(2, K.ndim(boxes))]
            new_classification_shape = [-1] + \
                [classification_shape[i] for i in range(2, K.ndim(classification) - 1)] + \
                [classification.get_shape()[-1]]
            new_other_shape = [[-1] + [o_s[i] for i in range(2, K.ndim(o))]
                               for o, o_s in zip(other, other_shape)]

            boxes = K.reshape(boxes, new_boxes_shape)
            classification = K.reshape(classification, new_classification_shape)
            other = [K.reshape(o, o_s) for o, o_s in zip(other, new_other_shape)]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes = args[0]
            classification = args[1]
            other = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[K.floatx(), K.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        if time_distributed:
            filtered_boxes = outputs[0]
            filtered_scores = outputs[1]
            filtered_labels = outputs[2]
            filtered_other = outputs[3:]

            final_boxes_shape = [boxes_shape[0], boxes_shape[1], self.max_detections, 4]
            final_scores_shape = [
                classification_shape[0],
                classification_shape[1],
                self.max_detections
            ]
            final_labels_shape = [
                classification_shape[0],
                classification_shape[1],
                self.max_detections
            ]
            final_others_shape = [[o[0], o[1], self.max_detections] +
                                  [o[i] for i in range(3, K.ndim(o))]
                                  for o in other_shape]

            filtered_boxes = K.reshape(filtered_boxes, final_boxes_shape)
            filtered_scores = K.reshape(filtered_scores, final_scores_shape)
            filtered_labels = K.reshape(filtered_labels, final_labels_shape)
            filtered_other = [K.reshape(o, o_s) for o, o_s in zip(filtered_other,
                                                                  final_others_shape)]

            outputs = [filtered_boxes, filtered_scores, filtered_labels] + filtered_other

        return outputs

    def compute_output_shape(self, input_shape):
        """Computes the output shapes given the input shapes.

        Args:
            input_shape : List of input shapes
                [boxes, classification, other[0], other[1], ...].

        Returns:
            list: List of tuples representing the output shapes:

            .. code-block:: python

                [
                    filtered_boxes.shape, filtered_scores.shape,
                    filtered_labels.shape, filtered_other[0].shape,
                    filtered_other[1].shape, ...
                ]
        """
        input_shape = [tensor_shape.TensorShape(insh) for insh in input_shape]
        # input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if len(input_shape[0]) == 3:
            return [
                (input_shape[0][0], self.max_detections, 4),
                (input_shape[1][0], self.max_detections),
                (input_shape[1][0], self.max_detections),
            ] + [
                tuple([input_shape[i][0], self.max_detections] +
                      list(input_shape[i][2:])) for i in range(2, len(input_shape))
            ]
        elif len(input_shape[0]) == 4:
            return [
                (input_shape[0][0], input_shape[0][1], self.max_detections, 4),
                (input_shape[1][0], input_shape[1][1], self.max_detections),
                (input_shape[1][0], input_shape[1][1], self.max_detections),
            ] + [
                tuple([input_shape[i][0], input_shape[i][1], self.max_detections] +
                      list(input_shape[i][3:])) for i in range(2, len(input_shape))
            ]

    def compute_mask(self, inputs, mask=None):
        """This is required in Keras when there is more than 1 output."""
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """Gets the configuration of this layer.

        Returns:
            Dictionary containing the parameters of this layer.
        """
        config = {
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
            'data_format': self.data_format
        }
        base_config = super(FilterDetections, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
