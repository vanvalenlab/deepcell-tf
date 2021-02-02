# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
import skimage as sk
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.framework import tensor_shape
# from cv2 import resize
from skimage.transform import resize

from deepcell.utils import compute_overlap


class AnchorParameters:
    """The parameteres that define how anchors are generated.

    Args:
        sizes (list): List of sizes to use.
            Each size corresponds to one feature level.
        strides (list): List of strides to use.
            Each stride correspond to one feature level.
        ratios (list): List of ratios to use per location in a feature map.
        scales (list): List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        """Get the number of anchors.

        Returns:
            int: the number of anchors
        """
        return len(self.ratios) * len(self.scales)


# The default anchor parameters.
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=[0.5, 1.0, 2.0],
    scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
)


def get_anchor_parameters(y):
    """Automatically determine appropriate backbone layers, pyarmid layers,
    and anchor parameters based on the annotated data.

    Args:
        y (numpy.array): Annotated data array (channels_last).

    Returns:
        tuple: Tuple of backbone layers, pyramid layers, and anchor parameters.
    """
    areas, aspects = [], []
    for batch in range(y.shape[0]):
        y_batch = y[batch, ..., 0]
        for prop in sk.measure.regionprops(y_batch):
            width = np.float(prop.bbox[2] - prop.bbox[0])
            height = np.float(prop.bbox[3] - prop.bbox[1])

            areas.append(width * height)
            aspects.append(width / height)

    aspects = np.log2(aspects)
    size_min = np.sqrt(np.percentile(areas, 2.5))

    size_max = np.sqrt(np.percentile(areas, 97.5))
    aspect_min = np.percentile(aspects, 2.5)
    aspect_max = np.percentile(aspects, 97.5)
    layer_min = np.maximum(np.floor(np.log2(size_min)) - 2, 1)
    layer_max = np.floor(np.log2(size_max)) - 2
    layers = np.arange(np.int(layer_min), np.int(layer_max) + 1)

    backbone_layers = ['C{}'.format(l) for l in layers]
    pyramid_layers = ['P{}'.format(l) for l in layers]

    sizes = 2.0 ** (layers + 2)
    strides = 2.0 ** (layers)
    ratios = 2.0 ** np.arange(np.int(aspect_min), np.int(aspect_max) + 1)
    scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    anchor_params = AnchorParameters(sizes=sizes, strides=strides,
                                     ratios=ratios, scales=scales)
    return backbone_layers, pyramid_layers, anchor_params


def generate_anchor_params(pyramid_levels, anchor_size_dict,
                           ratios=AnchorParameters.default.ratios,
                           scales=AnchorParameters.default.scales):
    """Get AnchorParameters for the given pyramid levels and anchor sizes.

    Args:
        pyramid_levels (list): List of layers to use as pyramid features
        anchor_size_dict (dict): dictionary of anchor sizes
        ratios (list): list of ratios
        scales (list): list of scales

    Returns:
        AnchorParameters: anchor configuration for the given
        pyramids and anchors
    """
    sizes = [anchor_size_dict[level] for level in pyramid_levels]
    strides = [2 ** int(level[1:]) for level in pyramid_levels]
    anchor_parameters = AnchorParameters(sizes, strides, ratios, scales)
    return anchor_parameters


def anchor_targets_bbox(anchors,
                        image_group,
                        annotations_group,
                        num_classes,
                        negative_overlap=0.4,
                        positive_overlap=0.5):
    """Generate anchor targets for bbox detection.

    Args:
        anchors (numpy.array): annotations of shape ``(N, 4)`` for
            ``(x1, y1, x2, y2)``.
        image_group (list): List of BGR images.
        annotations_group (list): List of annotations
            (np.array of shape ``(N, 5)`` for ``(x1, y1, x2, y2, label)``).
        num_classes (int): Number of classes to predict.
        mask_shape (numpy.array): If the image is padded with zeros,
            ``mask_shape`` can be used to mark the relevant part of the image.
        negative_overlap (float): IoU overlap for negative anchors
            (all anchors with overlap < ``negative_overlap`` are negative).
        positive_overlap (float): IoU overlap or positive anchors
            (all anchors with overlap > ``positive_overlap`` are positive).

    Returns:
        (numpy.array, numpy.array): The first numpy.array contains labels
        & anchor states with shape ``(batch_size, N, num_classes + 1)``,
        where ``N`` is the number of anchors for an image
        and the last column defines the anchor state
        (-1 for ignore, 0 for bg, 1 for fg).

        The second numpy.array contains bounding-box regression targets for
        an image & anchor states with shape ``(batch_size, N, 4 + 1)``,
        where ``N`` is the number of anchors for an image,
        the first 4 columns define regression targets for ``(x1, y1, x2, y2)``
        and the last column defines anchor states
        (-1 for ignore, 0 for bg, 1 for fg).
    """
    if len(image_group) != len(annotations_group):
        raise ValueError('Images and annotations must be the same size. '
                         'Got Image size = %s and Annotation size = %s' %
                         (len(image_group), len(annotations_group)))
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

    Args:
        anchors (numpy.array): annotations of shape ``(N, 4)`` for
            ``(x1, y1, x2, y2)``.
        annotations (numpy.array): shape ``(N, 5)`` for
            ``(x1, y1, x2, y2, label)``.
        negative_overlap (float): IoU overlap for negative anchors
            (all anchors with overlap < ``negative_overlap`` are negative).
        positive_overlap (float): IoU overlap or positive anchors
            (all anchors with overlap > ``positive_overlap`` are positive).

    Returns:
        tuple: (positive_indices, ignore_indices, argmax_overlaps_inds)
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """
    overlaps = compute_overlap(
        anchors.astype('float64'), annotations.astype('float64'))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def flatten_list(data):
    """Flatten a nested list of data.

    Args:
        data (list): list to be flattened

    Returns:
        list: flattened list.
    """
    results = []
    for rec in data:
        if isinstance(rec, list):
            results.extend(rec)
            results = flatten_list(results)
        else:
            results.append(rec)
    return results


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args:
        image_shape (tuple): The shape of the image.
        model (tensorflow.keras.Model): The model to use for computing how the image
            shape is transformed in the pyramid.

    Returns:
        dict: mapping of layer names to image shapes.
    """
    if isinstance(image_shape, tensor_shape.TensorShape):
        image_shape = tuple(image_shape.as_list())

    shape = {model.layers[0].name: (None,) + image_shape}

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in flatten_list([node.inbound_layers])]
            if not inputs:
                continue
            i = inputs[0] if len(inputs) == 1 else inputs
            computed_shape = layer.compute_output_shape(i)
            if isinstance(computed_shape, tensor_shape.TensorShape):
                computed_shape = computed_shape.as_list()
            shape[layer.name] = tuple(computed_shape)

    return shape


def make_shapes_callback(model):
    """Make a function for getting the shape of the pyramid levels.

    Args:
        model (tensorflow.keras.Model): model to get shapes of pyramid levels

    Returns:
        function: function that returns shapes
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        # input_shape = tensor_shape.TensorShape(input_shape).as_list()
        image_shapes = []
        for l in pyramid_levels:
            image_shape = shape['P{}'.format(l)][1:3]
            image_shape = tensor_shape.TensorShape(image_shape)
            image_shapes.append(image_shape)
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args:
        image_shape (tuple): The shape of the image.
        pyramid_levels (str[]): A list of what pyramid levels are used.

    Returns:
        list: image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = []
    for x in pyramid_levels:
        im_shape = (image_shape + 2 ** x - 1) // (2 ** x)
        im_shape = tensor_shape.TensorShape(im_shape)
        image_shapes.append(im_shape)
    return image_shapes


def anchors_for_shape(image_shape,
                      pyramid_levels=None,
                      anchor_params=None,
                      shapes_callback=None):
    """Generators anchors for a given shape.

    Args:
        image_shape (tuple): The shape of the image.
        pyramid_levels (int[]): List of ints representing which pyramids to use
            (defaults to ``[3, 4, 5, 6, 7]``).
        anchor_params (AnchorParameters): Struct containing anchor parameters.
            If ``None``, default values are used.
        shapes_callback (function): Function to call for getting the shape of
            the image at different pyramid levels.

    Returns:
        numpy.array: ``(N, 4)`` containing the ``(x1, y1, x2, y2)``
        anchor coordinates.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shape = tensor_shape.TensorShape(image_shape)
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        image_shape = image_shapes[idx].as_list()
        anchor_param = anchor_params.strides[idx]
        shifted_anchors = _shift(image_shape, anchor_param, anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def _shift(shape, stride, anchors):
    """Produce shifted anchors based on shape of the map and stride size.

    Args:
        shape (tuple): Shape to shift the anchors over.
        stride (int): Stride to shift the anchors with over the shape.
        anchors (numpy.array): The anchors to apply at each location.

    Returns:
        numpy.array: shifted anchors
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
    # cell C shifts (C, 1, 4) to get
    # shift anchors (C, A, 4)
    # reshape to (C*A, 4) shifted anchors
    A = anchors.shape[0]
    C = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, C, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((C * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.

    Args:
        base_size (int): base size of anchors
        ratios (list): list of ratios
        scales (list): list of scales

    Returns:
        numpy.array: generated anchors
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    ratios = np.array(ratios, dtype=K.floatx())
    scales = np.array(scales, dtype=K.floatx())

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
    """Compute bounding-box regression targets for an image.

    Args:
        anchors (numpy.array): locations of anchors
        gt_boxes (numpy.array): coordinates of bounding boxes
        mean (numpy.array): arithmetic mean
        std (numpy.array): standard deviation

    Raises:
        ValueError: mean is not a numpy.array
        ValueError: std is not a numpy.array
    """

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

    Args:
        boxes (numpy.array): shape ``(B, N, 4)``, where ``B`` is the batch
            size, ``N`` the number of boxes and 4 values for
            ``(x1, y1, x2, y2)``.
        deltas (numpy.array): same shape as boxes. These deltas
            ``(d_x1, d_y1, d_x2, d_y2)`` are a factor of the width/height.
        mean (numpy.array): The mean value used when computing deltas
            (defaults to ``[0, 0, 0, 0]``).
        std (numpy.array): The standard deviation used when computing deltas
            (defaults to ``[0.2, 0.2, 0.2, 0.2]``).

    Returns:
        numpy.array: same shape as boxes with deltas applied to each box.
        The mean and std are used during training to normalize the
        regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[..., 3] - boxes[..., 1]

    x1 = boxes[..., 0] + (deltas[..., 0] * std[0] + mean[0]) * width
    y1 = boxes[..., 1] + (deltas[..., 1] * std[1] + mean[1]) * height
    x2 = boxes[..., 2] + (deltas[..., 2] * std[2] + mean[2]) * width
    y2 = boxes[..., 3] + (deltas[..., 3] * std[3] + mean[3]) * height

    pred_boxes = K.stack([x1, y1, x2, y2], axis=-1)

    return pred_boxes


def shift(shape, stride, anchors):
    """Produce shifted anchors based on shape of the map and stride size.

    Args:
        shape (tuple): Shape to shift the anchors over.
        stride (int): Stride to shift the anchors with over the shape.
        anchors (numpy.array): The anchors to apply at each location.

    Returns:
        numpy.array: shifted anchors
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


def compute_iou(a, b):
    """Computes the IoU overlap of boxes in a and b.
    Args:
        a (numpy.array): ``(N, H, W)`` ndarray of float
        b (numpy.array): ``(K, H, W)`` ndarray of float
    Returns
        numpy.array: ``(N, K)`` ndarray of overlap between boxes
        and ``query_boxes``
    """
    intersection = np.zeros((a.shape[0], b.shape[0]))
    union = np.zeros((a.shape[0], b.shape[0]))
    for index, mask in enumerate(a):
        intersection[index, :] = np.sum(np.count_nonzero(np.logical_and(b, mask), axis=1), axis=1)
        union[index, :] = np.sum(np.count_nonzero(b + mask, axis=1), axis=1)

    return intersection / union


def overlap(a, b):
    """Computes the IoU overlap of boxes in a and b.

    Args:
        a (numpy.array): np.array of shape ``(N, 4)`` of boxes.
        b (numpy.array): np.array of shape ``(K, 4)`` of boxes.

    Returns:
        numpy.array: shape ``(N, K)`` of overlap between boxes
        from ``a`` and ``b``.
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = K.minimum(K.expand_dims(a[:, 2], axis=1), b[:, 2]) - \
        K.maximum(K.expand_dims(a[:, 0], axis=1), b[:, 0])
    ih = K.minimum(K.expand_dims(a[:, 3], axis=1), b[:, 3]) - \
        K.maximum(K.expand_dims(a[:, 1], axis=1), b[:, 1])

    iw = K.maximum(iw, 0)
    ih = K.maximum(ih, 0)

    ua = K.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + \
        area - iw * ih
    ua = K.maximum(ua, K.epsilon())

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall (numpy.array): The recall curve (list).
        precision (numpy.array): The precision curve (list).

    Returns:
        numpy.array: The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator,
                    model,
                    frames_per_batch=1,
                    score_threshold=0.05,
                    max_detections=100):
    """Get the detections from the model using the generator.

    The result is a list of lists such that the size is:

    .. code-block:: python

        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    Args:
        generator: The generator used to run images through the model.
        model (tensorflow.keras.Model): The model to run on the images.
        score_threshold (float): The score confidence threshold to use.
        max_detections (int): The maximum number of detections to use per image.

    Returns:
        list: The detections for each image in ``generator``.
    """
    all_detections = [[None for i in range(generator.num_classes)]
                      for j in range(generator.y.shape[0])]

    all_masks = [[None for i in range(generator.num_classes)]
                 for j in range(generator.y.shape[0])]

    if len(generator.x.shape) == 4:
        for i in range(generator.y.shape[0]):
            # raw_image = generator.load_image(i)
            # image = generator.preprocess_image(raw_image.copy())
            # image, scale = generator.resize_image(image)
            image = generator.x[i]

            # run network
            results = model.predict(np.expand_dims(image, axis=0))

            if generator.panoptic:
                num_semantic_outputs = len(generator.y_semantic_list)
                boxes = results[-num_semantic_outputs - 3]
                scores = results[-num_semantic_outputs - 2]
                labels = results[-num_semantic_outputs - 1]
                semantic = results[-num_semantic_outputs:]
                if generator.include_masks:
                    boxes = results[-num_semantic_outputs - 4]
                    scores = results[-num_semantic_outputs - 3]
                    labels = results[-num_semantic_outputs - 2]
                    masks = results[-num_semantic_outputs - 1]
                    semantic = results[-num_semantic_outputs]
            elif generator.include_masks:
                boxes = results[-4]
                scores = results[-3]
                labels = results[-2]
                masks = results[-1]
            else:
                boxes = results[-3]
                scores = results[-2]
                labels = results[-1]

            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > score_threshold)[0]

            # select those scores
            scores = scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_boxes = boxes[0, indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = labels[0, indices[scores_sort]]

            image_detections = np.concatenate([
                image_boxes,
                np.expand_dims(image_scores, axis=1),
                np.expand_dims(image_labels, axis=1)
            ], axis=1)

            # copy detections to all_detections
            for label in range(generator.num_classes):
                imd = image_detections[image_detections[:, -1] == label, :-1]
                all_detections[i][label] = imd

            if generator.include_masks:
                image_masks = masks[0, indices[scores_sort], :, :, image_labels]
                for label in range(generator.num_classes):
                    imm = image_masks[image_detections[:, -1] == label, ...]
                    all_masks[i][label] = imm

    if len(generator.x.shape) == 5:
        boxes_list = []
        scores_list = []
        labels_list = []
        masks_list = []

        for i in range(generator.y.shape[0]):
            for j in range(0, generator.y.shape[1], frames_per_batch):
                movie = generator.x[[i], j:j + frames_per_batch, ...]
                results = model.predict_on_batch(movie)

                if generator.panoptic:
                    # Add logic for networks that have semantic heads
                    pass
                else:
                    if generator.include_masks:
                        boxes = results[-4]
                        scores = results[-3]
                        labels = results[-2]
                        masks = results[-1]
                    else:
                        boxes, scores, labels = results[0:3]

                    for k in range(frames_per_batch):
                        boxes_list.append(boxes[0, k])
                        scores_list.append(scores[0, k])
                        labels_list.append(labels[0, k])
                        if generator.include_masks:
                            masks_list.append(masks[0, k])

        batch_boxes = np.stack(boxes_list, axis=0)
        batch_scores = np.stack(scores_list, axis=0)
        batch_labels = np.stack(labels_list, axis=0)

        all_detections = [[None for i in range(generator.num_classes)]
                          for j in range(batch_boxes.shape[0])]

        all_masks = [[None for i in range(generator.num_classes)]
                     for j in range(batch_boxes.shape[0])]

        for i in range(batch_boxes.shape[0]):
            boxes = batch_boxes[[i]]
            scores = batch_scores[[i]]
            labels = batch_labels[[i]]

            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > score_threshold)[0]

            # select those scores
            scores = scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_boxes = boxes[0, indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = labels[0, indices[scores_sort]]

            image_detections = np.concatenate([
                image_boxes,
                np.expand_dims(image_scores, axis=1),
                np.expand_dims(image_labels, axis=1)
            ], axis=1)

            # copy detections to all_detections
            for label in range(generator.num_classes):
                imd = image_detections[image_detections[:, -1] == label, :-1]
                all_detections[i][label] = imd

            if generator.include_masks:
                masks = np.expand_dims(masks_list[i], axis=0)
                image_masks = masks[0, indices[scores_sort], :, :, image_labels]
                for label in range(generator.num_classes):
                    imm = image_masks[image_detections[:, -1] == label, ...]
                    all_masks[i][label] = imm

    return all_detections, all_masks


def _get_annotations(generator, frames_per_batch=1):
    """Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:

    .. code-block:: python

        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        list: The annotations for each image in ``generator``.
    """

    if len(generator.x.shape) == 4:
        all_annotations = [[None for i in range(generator.num_classes)]
                           for j in range(generator.y.shape[0])]

        all_masks = [[None for i in range(generator.num_classes)]
                     for j in range(generator.y.shape[0])]

        for i in range(generator.y.shape[0]):
            # load the annotations
            annotations = generator.load_annotations(generator.y[i])

            if generator.include_masks:
                annotations['masks'] = np.stack(annotations['masks'], axis=0)

            # copy detections to all_annotations
            for label in range(generator.num_classes):
                imb = annotations['bboxes'][annotations['labels'] == label, :].copy()
                all_annotations[i][label] = imb
                if generator.include_masks:
                    imm = annotations['masks'][annotations['labels'] == label, ..., 0].copy()
                    all_masks[i][label] = imm

    if len(generator.x.shape) == 5:
        all_annotations = []
        all_masks = []
        for i in range(generator.y.shape[0]):
            for j in range(0, generator.y.shape[1], frames_per_batch):
                label_movie = generator.y[i, j:j + frames_per_batch, ...]
                for k in range(frames_per_batch):
                    annotations = generator.load_annotations(label_movie[k])

                    if generator.include_masks:
                        annotations['masks'] = np.stack(annotations['masks'], axis=0)

                    imb_list = [None for i in range(generator.num_classes)]
                    imm_list = [None for i in range(generator.num_classes)]
                    for label in range(generator.num_classes):
                        label_idx = annotations['labels'] == label
                        imb = annotations['bboxes'][label_idx, :].copy()
                        imb_list[label] = imb
                        if generator.include_masks:
                            imm = annotations['masks'][label_idx, ..., 0].copy()
                            imm_list[label] = imm
                    all_annotations.append(imb_list.copy())
                    all_masks.append(imm_list.copy())

    return all_annotations, all_masks


def evaluate(generator, model,
             iou_threshold=0.5,
             score_threshold=0.05,
             frames_per_batch=1,
             max_detections=100):
    """Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model (tensorflow.keras.Model): The model to evaluate.
        iou_threshold (float): The threshold used to consider when a detection
            is positive or negative.
        score_threshold (float): The score confidence threshold
            to use for detections.
        max_detections (int): The maximum number of detections to use per image.

    Returns:
        dict: A mapping of class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, _ = _get_detections(
        generator, model,
        frames_per_batch=frames_per_batch,
        score_threshold=score_threshold,
        max_detections=max_detections)
    all_annotations, _ = _get_annotations(generator, frames_per_batch)
    average_precisions = {}

    # process detections and annotations
    for label in range(generator.num_classes):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.y.shape[0]):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                # type `double` is required for `compute_overlap`
                annotations = annotations.astype('double')
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned]

                if max_overlap >= iou_threshold and assigned not in detected:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected.append(assigned)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives,
                                                np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def evaluate_mask(generator, model,
                  iou_threshold=0.5,
                  score_threshold=0.05,
                  max_detections=100,
                  frames_per_batch=1,
                  binarize_threshold=0.5):
    """Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model (tensorflow.keras.Model): The model to evaluate.
        iou_threshold (float): The threshold used to consider when a detection
            is positive or negative.
        score_threshold (float): The score confidence threshold
            to use for detections.
        max_detections (int): The maximum number of detections to use per image.
        binarize_threshold (float): Threshold to binarize the masks with.

    Returns:
        dict: A mapping of class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_masks = _get_detections(
        generator, model,
        frames_per_batch=frames_per_batch,
        score_threshold=score_threshold,
        max_detections=max_detections)
    all_annotations, all_gt_masks = _get_annotations(generator, frames_per_batch)
    average_precisions = {}

    # import pickle
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_masks, open('all_masks.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))
    # pickle.dump(all_gt_masks, open('all_gt_masks.pkl', 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.y.shape[0]):
            detections = all_detections[i][label]
            masks = all_masks[i][label]
            annotations = all_annotations[i][label]
            gt_masks = all_gt_masks[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d, mask in zip(detections, masks):
                box = d[:4].astype(int)
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                # resize to fit the box
                # mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))
                box_x = box[3] - box[1]
                box_y = box[2] - box[0]

                if frames_per_batch == 1:
                    mask = np.expand_dims(mask, axis=0)

                mask = resize(mask, (frames_per_batch, box_x, box_y))

                # binarize the mask
                mask = (mask > binarize_threshold).astype('uint8')

                # place mask in image frame
                mask_image = np.zeros(tuple([frames_per_batch] +
                                            list(gt_masks[0].shape)))

                mask_image[:, box[1]:box[3], box[0]:box[2]] = mask

                mask = mask_image

                for f in range(frames_per_batch):

                    overlaps = compute_iou(np.expand_dims(mask[f], axis=0), gt_masks)

                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and \
                       assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives,
                                                np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions
