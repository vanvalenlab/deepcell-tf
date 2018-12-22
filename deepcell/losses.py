# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
from tensorflow.python.keras import backend as K


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    Args:
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    Returns:
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def categorical_crossentropy(y_true, y_pred,
                             class_weights=None,
                             axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    
    Args:
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `y_pred` is expected to be the logits).
        from_logits: Boolean, whether `y_pred` is the
        result of a softmax, or is a tensor of logits.

    Returns:
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
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
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `y_pred` is expected to be the logits).
        from_logits: Boolean, whether `y_pred` is the
        result of a softmax, or is a tensor of logits.

    Returns:
        Output tensor.
    """
    if from_logits:
        raise Exception('weighted_categorical_crossentropy cannot take logits')
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    reduce_axis = [x for x in list(range(K.ndim(y_pred))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    y_true_cast = K.cast(y_true, K.floatx())
    total_sum = K.sum(y_true_cast)
    class_sum = K.sum(y_true_cast, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / K.cast_to_floatx(n_classes) * tf.divide(total_sum, class_sum + 1.)
    return - K.sum((y_true * K.log(y_pred) * class_weights), axis=axis)


def sample_categorical_crossentropy(y_true,
                                    y_pred,
                                    class_weights=None,
                                    axis=None,
                                    from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    Only the sampled pixels defined by y_true = 1 are used to compute the cross entropy

    Args:
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `y_pred` is expected to be the logits).
        from_logits: Boolean, whether `y_pred` is the
        result of a softmax, or is a tensor of logits.

    Returns:
        Output tensor.
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
        _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        if class_weights is None:
            return - K.sum(y_true * K.log(y_pred), axis=axis)
        return - K.sum((y_true * K.log(y_pred) * class_weights), axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def dice_loss(y_true, y_pred, smooth=1):
    """Computes the dice loss

    Args:
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax

    Returns:
        dice loss
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


    """Computes the discriminative instance loss
def discriminative_instance_loss(y_true, y_pred,
                                 delta_v=0.5,
                                 delta_d=1.5,
                                 gamma=1e-3):

    Args:
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor of the vector embedding

    Returns:
        loss
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
    n_pixels = K.cast(tf.count_nonzero(y_true, axis=axes), dtype=K.floatx()) + K.epsilon()
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
    diag = K.constant(0, dtype=K.floatx()) * tf.diag_part(L_dist_2)
    L_dist_3 = tf.matrix_set_diag(L_dist_2, diag)
    L_dist = K.mean(L_dist_3)

    # Compute regularization loss
    L_reg = gamma * temp_norm(mu)
    L = L_var + L_dist + K.mean(L_reg)

    return L


def weighted_focal_loss(y_true, y_pred, n_classes=3, gamma=2., axis=None, from_logits=False):
    """Computes focal loss with class weights computed on the fly
    
    Args:
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax

    Returns:
        focal loss
    """
    if from_logits:
        raise Exception('weighted_focal_loss cannot take logits')
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else K.ndim(y_pred) - 1
    reduce_axis = [x for x in list(range(K.ndim(y_pred))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / K.sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    y_true_cast = K.cast(y_true, K.floatx())
    total_sum = K.sum(y_true_cast)
    class_sum = K.sum(y_true_cast, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / K.cast_to_floatx(n_classes) * tf.divide(total_sum, class_sum + 1.)
    temp_loss = (K.pow(1. - y_pred, gamma) * K.log(y_pred) * class_weights)
    focal_loss = - K.sum(y_true * temp_loss, axis=axis)
    return focal_loss
