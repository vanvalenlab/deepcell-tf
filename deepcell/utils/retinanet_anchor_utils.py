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
"""Utilities for generating RetinaNet Anchors"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

try:
    from deepcell.utils.compute_overlap import compute_overlap
except ImportError:
    compute_overlap = None


class AnchorParameters:
    """The parameteres that define how anchors are generated.

    Args
        sizes: List of sizes to use. Each size corresponds to one feature level.
        strides: List of strides to use. Each stride correspond to one feature level.
        ratios: List of ratios to use per location in a feature map.
        scales: List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


# The default anchor parameters.
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array([0.5, 1, 2], K.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], K.floatx()),
)


def anchor_targets_bbox(anchors,
                        image_group,
                        annotations_group,
                        num_classes,
                        negative_overlap=0.4,
                        positive_overlap=0.5):
    """Generate anchor targets for bbox detection.

    Args:
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations
            (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used
            to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with
            overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with
            overlap > positive_overlap are positive).

    Returns:
        labels_batch: batch that contains labels & anchor states
            (np.array of shape (batch_size, N, num_classes + 1), where N is the
            number of anchors for an image and the last column defines the
            anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets
            for an image & anchor states
            (np.array of shape (batch_size, N, 4 + 1), where N is the number of
            anchors for an image, the first 4 columns define regression targets
            for (x1, y1, x2, y2) and the last column defines anchor states
            (-1 for ignore, 0 for bg, 1 for fg).
    """
    if len(image_group) != len(annotations_group):
        raise ValueError('Images and annotations must be the same size. '
                         'Got Image size = %s and Annotation size = %s' %
                         len((image_group), len(annotations_group)))
    elif len(annotations_group) == 0:
        raise ValueError('No data received to compute anchor targets.')
    for annotations in annotations_group:
        if 'bboxes' not in annotations:
            raise ValueError('Annotations should contain bboxes.')
        if 'labels' not in annotations:
            raise ValueError('Annotations should contain labels.')

    regress_shape = (image_group.shape[0], anchors.shape[0], 4 + 1)
    labels_shape = (image_group.shape[0], anchors.shape[0], num_classes + 1)

    regression_batch = np.zeros(regress_shape, dtype=K.floatx())
    labels_batch = np.zeros(labels_shape, dtype=K.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(
                anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1] = -1
            labels_batch[index, positive_indices, -1] = 1

            regression_batch[index, ignore_indices, -1] = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[
                index,
                positive_indices,
                annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)
            ] = 1

            regression_batch[index, :, :-1] = bbox_transform(
                anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack([
                (anchors[:, 0] + anchors[:, 2]) / 2,
                (anchors[:, 1] + anchors[:, 3]) / 2
            ]).T

            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1],
                                    anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1

    return regression_batch, labels_batch


def compute_gt_annotations(anchors,
                           annotations,
                           negative_overlap=0.4,
                           positive_overlap=0.5):
    """Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors
            (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors
            (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """
    if compute_overlap is None:
        raise ImportError('To use `compute_overlap`, the C extensions must be '
                          'built using `python setup.py build_ext --inplace`')
    overlaps = compute_overlap(
        anchors.astype(np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is
            transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {model.layers[0].name: (None,) + image_shape}

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            i = inputs[0] if len(inputs) == 1 else inputs
            shape[layer.name] = layer.compute_output_shape(i)

    return shape


def make_shapes_callback(model):
    """Make a function for getting the shape of the pyramid levels.
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape['P{}'.format(l)][1:3] for l in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(image_shape,
                      pyramid_levels=None,
                      anchor_params=None,
                      shapes_callback=None):
    """Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use
            (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters.
            If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image
            at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates
            for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def _shift(shape, stride, anchors):
    """Produce shifted anchors based on shape of the map and stride size.

    Args
        shape: Shape to shift the anchors over.
        stride: Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4), dtype=K.floatx())

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. '
                         'Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. '
                         'Received: {}'.format(type(std)))

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was
    previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator.
    They are unnormalized in this function and then applied to the boxes.

    Args
        boxes: np.array of shape (B, N, 4), where B is the batch size,
               N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas
                (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean: The mean value used when computing deltas
              (defaults to [0, 0, 0, 0]).
        std: The standard deviation used when computing deltas
             (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes with deltas applied to each box.
        The mean and std are used during training to normalize the
        regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = K.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    """Produce shifted anchors based on shape of the map and stride size.

    Args
        shape: Shape to shift the anchors over.
        stride: Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.

    Returns:
        shifted anchors
    """
    shift_x = (K.arange(0, shape[1], dtype=K.floatx()) +
               K.constant(0.5, dtype=K.floatx())) * stride
    shift_y = (K.arange(0, shape[0], dtype=K.floatx()) +
               K.constant(0.5, dtype=K.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = K.reshape(shift_x, [-1])
    shift_y = K.reshape(shift_y, [-1])

    shifts = K.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = K.transpose(shifts)
    number_of_anchors = K.shape(anchors)[0]

    k = K.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifts = K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
    shifted_anchors = K.reshape(anchors, [1, number_of_anchors, 4]) + shifts
    shifted_anchors = K.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors
