# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    n_classes = K.cast(n_classes, y_pred.dtype)
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    reduce_axis = [x for x in list(range(K.ndim(y_pred))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    total_sum = K.sum(y_true)
    class_sum = K.sum(y_true, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / n_classes * tf.divide(total_sum, class_sum + 1.)
    return - K.sum((y_true * K.log(y_pred) * class_weights), axis=axis)


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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

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
    n_pixels = K.cast(nonzeros, dtype=y_pred.dtype) + K.epsilon()
    n_pixels_expand = K.expand_dims(n_pixels, axis=1) + K.epsilon()
    mu = tf.divide(cells_summed, n_pixels_expand)

    delta_v = K.constant(delta_v, dtype=y_pred.dtype)
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
    L_dist_2 = K.square(K.relu(K.constant(2 * delta_d, dtype=y_pred.dtype) - L_dist_1))
    diag = K.constant(0, dtype=y_pred.dtype) * tf.linalg.diag_part(L_dist_2)
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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    n_classes = K.cast(n_classes, y_pred.dtype)
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    reduce_axis = [x for x in list(range(K.ndim(y_pred))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    total_sum = K.sum(y_true)
    class_sum = K.sum(y_true, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / n_classes * tf.divide(total_sum, class_sum + 1.)
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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

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
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1

    # compute the focal loss
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)

    return K.sum(cls_loss, axis=axis)
