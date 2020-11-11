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
"""Custom loss functions for DeepCell"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras import backend as K

from deepcell.utils.retinanet_anchor_utils import overlap


def categorical_crossentropy(y_true, y_pred, class_weights=None, axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.

    Args:
        y_true: A tensor of the same shape as output.
        y_pred: A tensor resulting from a softmax
            (unless ``from_logits`` is ``True``, in which
            case ``y_pred`` is expected to be the logits).
        from_logits: Boolean, whether ``y_pred`` is the
            result of a softmax, or is a tensor of logits.

    Returns:
        tensor: Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        if class_weights is None:
            return - K.sum(y_true * K.log(y_pred), axis=axis)
        return - K.sum((y_true * K.log(y_pred) * class_weights), axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def weighted_categorical_crossentropy(y_true, y_pred,
                                      n_classes=3, axis=None,
                                      from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    Automatically computes the class weights from the target image and uses
    them to weight the cross entropy

    Args:
        y_true: A tensor of the same shape as ``y_pred``.
        y_pred: A tensor resulting from a softmax
            (unless ``from_logits`` is ``True``, in which
            case ``y_pred`` is expected to be the logits).
        from_logits: Boolean, whether ``y_pred`` is the
            result of a softmax, or is a tensor of logits.

    Returns:
        tensor: Output tensor.
    """
    if from_logits:
        raise Exception('weighted_categorical_crossentropy cannot take logits')
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    reduce_axis = [x for x in list(range(K.ndim(y_pred))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    y_true_cast = K.cast(y_true, K.floatx())
    total_sum = K.sum(y_true_cast)
    class_sum = K.sum(y_true_cast, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / K.cast_to_floatx(n_classes) * tf.divide(total_sum, class_sum + 1.)
    return - K.sum((y_true_cast * K.log(y_pred) * class_weights), axis=axis)


def sample_categorical_crossentropy(y_true,
                                    y_pred,
                                    class_weights=None,
                                    axis=None,
                                    from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    Only the sampled pixels defined by y_true = 1 are used to compute the
    cross entropy.

    Args:
        y_true: A tensor of the same shape as ``y_pred``.
        y_pred: A tensor resulting from a softmax
            (unless ``from_logits`` is ``True``, in which
            case ``y_pred`` is expected to be the logits).
        from_logits: Boolean, whether ``y_pred`` is the
            result of a softmax, or is a tensor of logits.

    Returns:
        tensor: Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    if not from_logits:
        # scale preds so that the class probabilities of each sample sum to 1
        y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)

        # Multiply with mask so that only the sampled pixels are used
        y_pred = y_pred * y_true

        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        if class_weights is None:
            return - K.sum(y_true * K.log(y_pred), axis=axis)
        return - K.sum((y_true * K.log(y_pred) * class_weights), axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def dice_loss(y_true, y_pred, smooth=1):
    """Dice coefficient loss between an output tensor and a target tensor.

    Args:
        y_true: A tensor of the same shape as ``y_pred``.
        y_pred: A tensor resulting from a softmax

    Returns:
        tensor: Output tensor.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def discriminative_instance_loss(y_true, y_pred,
                                 delta_v=0.5,
                                 delta_d=1.5,
                                 gamma=1e-3):
    """Discriminative loss between an output tensor and a target tensor.

    Args:
        y_true: A tensor of the same shape as ``y_pred``.
        y_pred: A tensor of the vector embedding

    Returns:
        tensor: Output tensor.
    """

    def temp_norm(ten, axis=None):
        if axis is None:
            axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(ten) - 1
        return K.sqrt(K.epsilon() + K.sum(K.square(ten), axis=axis))

    rank = K.ndim(y_pred)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else rank - 1
    axes = [x for x in list(range(rank)) if x != channel_axis]

    # Compute variance loss
    cells_summed = tf.tensordot(y_true, y_pred, axes=[axes, axes])
    nonzeros = tf.math.count_nonzero(y_true, axis=axes)
    n_pixels = K.cast(nonzeros, dtype=K.floatx()) + K.epsilon()
    n_pixels_expand = K.expand_dims(n_pixels, axis=1) + K.epsilon()
    mu = tf.divide(cells_summed, n_pixels_expand)

    delta_v = K.constant(delta_v, dtype=K.floatx())
    mu_tensor = tf.tensordot(y_true, mu, axes=[[channel_axis], [0]])
    L_var_1 = y_pred - mu_tensor
    L_var_2 = K.square(K.relu(temp_norm(L_var_1) - delta_v))
    L_var_3 = tf.tensordot(L_var_2, y_true, axes=[axes, axes])
    L_var_4 = tf.divide(L_var_3, n_pixels)
    L_var = K.mean(L_var_4)

    # Compute distance loss
    mu_a = K.expand_dims(mu, axis=0)
    mu_b = K.expand_dims(mu, axis=1)

    diff_matrix = tf.subtract(mu_b, mu_a)
    L_dist_1 = temp_norm(diff_matrix)
    L_dist_2 = K.square(K.relu(K.constant(2 * delta_d, dtype=K.floatx()) - L_dist_1))
    diag = K.constant(0, dtype=K.floatx()) * tf.linalg.diag_part(L_dist_2)
    L_dist_3 = tf.linalg.set_diag(L_dist_2, diag)
    L_dist = K.mean(L_dist_3)

    # Compute regularization loss
    L_reg = gamma * temp_norm(mu)
    L = L_var + L_dist + K.mean(L_reg)

    return L


def weighted_focal_loss(y_true, y_pred, n_classes=3, gamma=2., axis=None, from_logits=False):
    """Focal loss between an output tensor and a target tensor.
    Automatically computes the class weights from the target image and uses
    them to weight the cross entropy

    Args:
        y_true: A tensor of the same shape as ``y_pred``.
        y_pred: A tensor resulting from a softmax
            (unless ``from_logits`` is ``True``, in which
            case ``y_pred`` is expected to be the logits).
        from_logits: Boolean, whether ``y_pred`` is the
            result of a softmax, or is a tensor of logits.

    Returns:
        tensor: Output tensor.
    """
    if from_logits:
        raise Exception('weighted_focal_loss cannot take logits')
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    reduce_axis = [x for x in list(range(K.ndim(y_pred))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    y_true_cast = K.cast(y_true, K.floatx())
    total_sum = K.sum(y_true_cast)
    class_sum = K.sum(y_true_cast, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / K.cast_to_floatx(n_classes) * tf.divide(total_sum, class_sum + 1.)
    temp_loss = (K.pow(1. - y_pred, gamma) * K.log(y_pred) * class_weights)
    focal_loss = - K.sum(y_true * temp_loss, axis=axis)
    return focal_loss


def smooth_l1(y_true, y_pred, sigma=3.0, axis=None):
    """Compute the smooth L1 loss of ``y_pred`` w.r.t. ``y_true``.

    Args:
        y_true: Tensor from the generator of shape ``(B, N, 5)``.
            The last value for each box is the state of the anchor
            (ignore, negative, positive).
        y_pred: Tensor from the network of shape ``(B, N, 4)``.
        sigma: The point where the loss changes from L2 to L1.

    Returns:
        The smooth L1 loss of ``y_pred`` w.r.t. ``y_true``.
    """
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1

    sigma_squared = sigma ** 2

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = K.abs(y_true - y_pred)  # |y - f(x)|

    regression_loss = tf.where(
        K.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * K.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared)
    return K.sum(regression_loss, axis=axis)


def focal(y_true, y_pred, alpha=0.25, gamma=2.0, axis=None):
    """Compute the focal loss given the target tensor and the predicted tensor.

    As defined in https://arxiv.org/abs/1708.02002

    Args:
        y_true: Tensor of target data with shape ``(B, N, num_classes)``.
        y_pred: Tensor of predicted data with shape ``(B, N, num_classes)``.
        alpha: Scale the focal weight with ``alpha``.
        gamma: Take the power of the focal weight with ``gamma``.

    Returns:
        float: The focal loss of ``y_pred`` w.r.t. ``y_true``.
    """
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1

    # compute the focal loss
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)

    return K.sum(cls_loss, axis=axis)


"""
Retinanet losses
"""


class RetinaNetLosses(object):
    def __init__(self, sigma=3.0, alpha=0.25, gamma=2.0,
                 iou_threshold=0.5, fdl_iou_threshold=0.5,
                 mask_size=(28, 28),
                 parallel_iterations=32):
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.iou_threshold = iou_threshold
        self.fdl_iou_threshold = fdl_iou_threshold
        self.mask_size = mask_size
        self.parallel_iterations = parallel_iterations

    def regress_loss(self, y_true, y_pred):
        # separate target and state
        regression = y_pred
        regression_target = y_true[..., :-1]
        anchor_state = y_true[..., -1]

        # filter out "ignore" anchors
        indices = tf.where(K.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute the loss
        loss = smooth_l1(regression_target, regression, sigma=self.sigma)

        # compute the normalizer: the number of positive anchors
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())

        return K.sum(loss) / normalizer

    def classification_loss(self, y_true, y_pred):
        # TODO: try weighted_categorical_crossentropy
        labels = y_true[..., :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_true[..., -1]

        classification = y_pred
        # filter out "ignore" anchors
        indices = tf.where(K.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the loss
        loss = focal(labels, classification, alpha=self.alpha, gamma=self.gamma)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return K.sum(loss) / normalizer

    def mask_loss(self, y_true, y_pred):
        def _mask_conditional(y_true, y_pred):
            # if there are no masks annotations, return 0; else, compute the masks loss
            return tf.cond(
                K.any(K.equal(K.shape(y_true), 0)),
                lambda: K.cast_to_floatx(0.0),
                lambda: _mask_batch(y_true, y_pred,
                                    iou_threshold=self.iou_threshold,
                                    mask_size=self.mask_size,
                                    parallel_iterations=self.parallel_iterations)
            )

        def _mask_batch(y_true, y_pred,
                        iou_threshold=0.5,
                        mask_size=(28, 28),
                        parallel_iterations=32):
            if K.ndim(y_pred) == 4:
                y_pred_shape = tf.shape(y_pred)
                new_y_pred_shape = [y_pred_shape[0] * y_pred_shape[1],
                                    y_pred_shape[2], y_pred_shape[3]]
                y_pred = tf.reshape(y_pred, new_y_pred_shape)

                y_true_shape = tf.shape(y_true)
                new_y_true_shape = [y_true_shape[0] * y_true_shape[1],
                                    y_true_shape[2], y_true_shape[3]]
                y_true = tf.reshape(y_true, new_y_true_shape)

            # split up the different predicted blobs
            boxes = y_pred[:, :, :4]
            masks = y_pred[:, :, 4:]

            # split up the different blobs
            annotations = y_true[:, :, :5]
            width = K.cast(y_true[0, 0, 5], dtype='int32')
            height = K.cast(y_true[0, 0, 6], dtype='int32')
            masks_target = y_true[:, :, 7:]

            # reshape the masks back to their original size
            masks_target = K.reshape(masks_target, (K.shape(masks_target)[0],
                                                    K.shape(masks_target)[1],
                                                    height, width))
            masks = K.reshape(masks, (K.shape(masks)[0], K.shape(masks)[1],
                                      mask_size[0], mask_size[1], -1))

            def _mask(args):
                boxes = args[0]
                masks = args[1]
                annotations = args[2]
                masks_target = args[3]

                return compute_mask_loss(
                    boxes,
                    masks,
                    annotations,
                    masks_target,
                    width,
                    height,
                    iou_threshold=iou_threshold,
                    mask_size=mask_size,
                )

            mask_batch_loss = tf.map_fn(
                _mask,
                elems=[boxes, masks, annotations, masks_target],
                dtype=K.floatx(),
                parallel_iterations=parallel_iterations
            )

            return K.mean(mask_batch_loss)

        return _mask_conditional(y_true, y_pred)

    def final_detection_loss(self, y_true, y_pred):
        def _fd_conditional(y_true, y_pred):
            # if there are no masks annotations, return 0; else, compute fdl loss
            return tf.cond(
                K.any(K.equal(K.shape(y_true), 0)),
                lambda: K.cast_to_floatx(0.0),
                lambda: _fd_batch(y_true, y_pred,
                                  iou_threshold=self.fdl_iou_threshold,
                                  parallel_iterations=self.parallel_iterations))

        def _fd_batch(y_true, y_pred, iou_threshold=0.75, parallel_iterations=32):
            if K.ndim(y_pred) == 4:
                y_pred_shape = tf.shape(y_pred)
                new_y_pred_shape = [y_pred_shape[0] * y_pred_shape[1],
                                    y_pred_shape[2], y_pred_shape[3]]
                y_pred = tf.reshape(y_pred, new_y_pred_shape)

                y_true_shape = tf.shape(y_true)
                new_y_true_shape = [y_true_shape[0] * y_true_shape[1],
                                    y_true_shape[2], y_true_shape[3]]
                y_true = tf.reshape(y_true, new_y_true_shape)

            # split up the different predicted blobs
            boxes = y_pred[:, :, :4]
            scores = y_pred[:, :, 4:5]

            # split up the different blobs
            annotations = y_true[:, :, :5]

            def _fd(args):
                boxes = args[0]
                scores = args[1]
                annotations = args[2]

                return compute_fd_loss(
                    boxes,
                    scores,
                    annotations,
                    iou_threshold=iou_threshold)

            fd_batch_loss = tf.map_fn(
                _fd,
                elems=[boxes, scores, annotations],
                dtype=K.floatx(),
                parallel_iterations=parallel_iterations)

            return K.mean(fd_batch_loss)

        return _fd_conditional(y_true, y_pred)


def compute_mask_loss(boxes,
                      masks,
                      annotations,
                      masks_target,
                      width,
                      height,
                      iou_threshold=0.5,
                      mask_size=(28, 28)):
    """compute overlap of boxes with annotations"""
    iou = overlap(boxes, annotations)
    argmax_overlaps_inds = K.argmax(iou, axis=1)
    max_iou = K.max(iou, axis=1)

    # filter those with IoU > 0.5
    indices = tf.where(K.greater_equal(max_iou, iou_threshold))
    boxes = tf.gather_nd(boxes, indices)
    masks = tf.gather_nd(masks, indices)
    argmax_overlaps_inds = K.cast(tf.gather_nd(argmax_overlaps_inds, indices), 'int32')
    labels = K.cast(K.gather(annotations[:, 4], argmax_overlaps_inds), 'int32')

    # make normalized boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    boxes = K.stack([
        y1 / (K.cast(height, dtype=K.floatx()) - 1),
        x1 / (K.cast(width, dtype=K.floatx()) - 1),
        (y2 - 1) / (K.cast(height, dtype=K.floatx()) - 1),
        (x2 - 1) / (K.cast(width, dtype=K.floatx()) - 1),
    ], axis=1)

    # crop and resize masks_target
    # append a fake channel dimension
    masks_target = K.expand_dims(masks_target, axis=3)
    masks_target = tf.image.crop_and_resize(
        masks_target,
        boxes,
        argmax_overlaps_inds,
        mask_size
    )
    masks_target = masks_target[:, :, :, 0]  # remove fake channel dimension

    # gather the predicted masks using the annotation label
    masks = tf.transpose(masks, (0, 3, 1, 2))
    label_indices = K.stack([tf.range(K.shape(labels)[0]), labels], axis=1)

    masks = tf.gather_nd(masks, label_indices)

    # compute mask loss
    mask_loss = K.binary_crossentropy(masks_target, masks)
    normalizer = K.shape(masks)[0] * K.shape(masks)[1] * K.shape(masks)[2]
    normalizer = K.maximum(K.cast(normalizer, K.floatx()), 1)
    mask_loss = K.sum(mask_loss) / normalizer

    return mask_loss


def compute_fd_loss(boxes, scores, annotations, iou_threshold=0.75):
    """compute the overlap of boxes with annotations"""
    iou = overlap(boxes, annotations)

    max_iou = K.max(iou, axis=1, keepdims=True)
    targets = K.cast(K.greater_equal(max_iou, iou_threshold), K.floatx())

    # compute the loss
    loss = focal(targets, scores)  # alpha=self.alpha, gamma=self.gamma)

    # compute the normalizer: the number of cells present in the image
    normalizer = K.cast(K.shape(annotations)[0], K.floatx())
    normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

    return K.sum(loss) / normalizer
