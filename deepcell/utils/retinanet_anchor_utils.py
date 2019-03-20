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
# from cv2 import resize
from skimage.transform import resize
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import tensor_shape

try:
    from deepcell.utils.compute_overlap import compute_overlap
except ImportError:
    compute_overlap = None


class AnchorParameters:
    """The parameteres that define how anchors are generated.

    Args:
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
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors
            (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors
            (all anchors with overlap > positive_overlap are positive).

    Returns:
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """
    if compute_overlap is None:
        raise ImportError('To use `compute_overlap`, the C extensions must be '
                          'built using `python setup.py build_ext --inplace`')
    overlaps = compute_overlap(
        anchors.astype('float64'), annotations.astype('float64'))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args:
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is
            transformed in the pyramid.

    Returns:
        A dictionary mapping layer names to image shapes.
    """
    if isinstance(image_shape, tensor_shape.TensorShape):
        image_shape = tuple(image_shape.as_list())

    shape = {model.layers[0].name: (None,) + image_shape}

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
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
        model: keras.Model to get shapes of pyramid levels
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
        image_shape: The shape of the image.
        pyramid_levels: A list of what pyramid levels are used.

    Returns:
        A list of image shapes at each pyramid level.
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
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use
            (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters.
            If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image
            at different pyramid levels.

    Returns:
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates
            for the anchors.
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

    Args:
        boxes: np.array of shape (B, N, 4), where B is the batch size,
               N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas
                (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean: The mean value used when computing deltas
              (defaults to [0, 0, 0, 0]).
        std: The standard deviation used when computing deltas
             (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns:
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

    Args:
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


def compute_iou(a, b):
    """Computes the IoU overlap of boxes in a and b.
    Args:
        a: (N, H, W) ndarray of float
        b: (K, H, W) ndarray of float
    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    intersection = np.zeros((a.shape[0], b.shape[0]))
    union = np.zeros((a.shape[0], b.shape[0]))
    for index, mask in enumerate(a):
        intersection[index, :] = np.sum(np.count_nonzero(b == mask, axis=1), axis=1)
        union[index, :] = np.sum(np.count_nonzero(b + mask, axis=1), axis=1)

    return intersection / union


def overlap(a, b):
    """Computes the IoU overlap of boxes in a and b.

    Args:
        a: np.array of shape (N, 4) of boxes.
        b: np.array of shape (K, 4) of boxes.

    Returns:
        A np.array of shape (N, K) of overlap between boxes from a and b.
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
        recall: The recall curve (list).
        precision: The precision curve (list).
    Returns:
        The average precision as computed in py-faster-rcnn.
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
                    score_threshold=0.05,
                    max_detections=100):
    """Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
    Returns:
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes)]
                      for j in range(generator.y.shape[0])]

    all_masks = [[None for i in range(generator.num_classes)]
                 for j in range(generator.y.shape[0])]

    for i in range(generator.y.shape[0]):
        # raw_image = generator.load_image(i)
        # image = generator.preprocess_image(raw_image.copy())
        # image, scale = generator.resize_image(image)
        image = generator.x[i]

        # run network
        results = model.predict_on_batch(np.expand_dims(image, axis=0))
        if generator.include_masks:
            boxes = results[-4]
            scores = results[-3]
            labels = results[-2]
            masks = results[-1]
        else:
            boxes, scores, labels = results[:3]

        # correct boxes for image scale
        # boxes = boxes / scale

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

    return all_detections, all_masks


def _get_annotations(generator):
    """Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]

    Args:
        generator : The generator used to retrieve ground truth annotations.
    Returns:
        A list of lists containing the annotations for each image in the generator.
    """
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

    return all_annotations, all_masks


def evaluate(generator, model,
             iou_threshold=0.5,
             score_threshold=0.05,
             max_detections=100):
    """Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection
            is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
    Returns:
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, _ = _get_detections(
        generator, model,
        score_threshold=score_threshold,
        max_detections=max_detections)

    all_annotations, _ = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

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
                  binarize_threshold=0.5):
    """Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is
            positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        binarize_threshold: Threshold to binarize the masks with.

    Returns:
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_masks = _get_detections(
        generator, model,
        score_threshold=score_threshold,
        max_detections=max_detections)
    all_annotations, all_gt_masks = _get_annotations(generator)
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
                mask = resize(mask, (box[3] - box[1], box[2] - box[0]))

                # binarize the mask
                mask = (mask > binarize_threshold).astype('uint8')

                # place mask in image frame
                mask_image = np.zeros_like(gt_masks[0])
                mask_image[box[1]:box[3], box[0]:box[2]] = mask
                mask = mask_image

                overlaps = compute_iou(np.expand_dims(mask, axis=0), gt_masks)

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
