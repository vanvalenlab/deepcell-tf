# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Custom loss functions
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def categorical_crossentropy(y_true, y_pred, class_weights=None, axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        y_true: A tensor of the same shape as `output`.
        y_pred: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `y_pred` is expected to be the logits).
        from_logits: Boolean, whether `y_pred` is the
        result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else len(y_pred.get_shape()) - 1
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        if class_weights is None:
            return - tf.reduce_sum(y_true * tf.log(y_pred), axis=axis)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights), axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def weighted_categorical_crossentropy(y_true, y_pred, n_classes=3, axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    Automatically computes the class weights from the target image and uses
    them to weight the cross entropy
    # Arguments
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `y_pred` is expected to be the logits).
        from_logits: Boolean, whether `y_pred` is the
        result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if from_logits:
        raise Exception('weighted_categorical_crossentropy cannot take logits')
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else len(y_pred.get_shape()) - 1
    reduce_axis = [x for x in list(range(len(y_pred.get_shape()))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    y_true_cast = tf.cast(y_true, K.floatx())
    total_sum = tf.reduce_sum(y_true_cast)
    class_sum = tf.reduce_sum(y_true_cast, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / np.float(n_classes) * tf.divide(total_sum, class_sum + 1.)
    return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights), axis=axis)


def sample_categorical_crossentropy(y_true, y_pred, class_weights=None, axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor. Only the sampled
    pixels defined by y_true = 1 are used to compute the cross entropy
    # Arguments
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `y_pred` is expected to be the logits).
        from_logits: Boolean, whether `y_pred` is the
        result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else len(y_pred.get_shape()) - 1
    if not from_logits:
        # scale preds so that the class probabilities of each sample sum to 1
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)

        # Multiply with mask so that only the sampled pixels are used
        y_pred = tf.multiply(y_pred, y_true)

        # manual computation of crossentropy
        _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        if class_weights is None:
            return - tf.reduce_sum(y_true * tf.log(y_pred), axis=axis)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), class_weights), axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def dice_loss(y_true, y_pred, smooth=1):
    """Computes the dice loss
    # Arguments:
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def discriminative_instance_loss(y_true, y_pred, delta_v=0.5, delta_d=1.5, order=2, gamma=1e-3):
    """Computes the discriminative instance loss
    # Arguments:
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor of the vector embedding
    """

    def temp_norm(ten, axis=-1):
        return tf.sqrt(K.epsilon() + tf.reduce_sum(tf.square(ten), axis=axis))

    channel_axis = 1 if K.image_data_format() == 'channels_first' else len(y_pred.get_shape()) - 1
    other_axes = [x for x in list(range(len(y_pred.get_shape()))) if x != channel_axis]

    # Compute variance loss
    cells_summed = tf.tensordot(y_true, y_pred, axes=[other_axes, other_axes])
    n_pixels = tf.cast(tf.count_nonzero(y_true, axis=other_axes), dtype=K.floatx()) + K.epsilon()
    n_pixels_expand = tf.expand_dims(n_pixels, axis=1) + K.epsilon()
    mu = tf.divide(cells_summed, n_pixels_expand)

    mu_tensor = tf.tensordot(y_true, mu, axes=[[channel_axis], [0]])
    L_var_1 = y_pred - mu_tensor
    L_var_2 = tf.square(tf.nn.relu(temp_norm(L_var_1, axis=channel_axis) - tf.constant(delta_v, dtype=K.floatx())))
    L_var_3 = tf.tensordot(L_var_2, y_true, axes=[other_axes, other_axes])
    L_var_4 = tf.divide(L_var_3, n_pixels)
    L_var = tf.reduce_mean(L_var_4)

    # Compute distance loss
    mu_a = tf.expand_dims(mu, axis=0)
    mu_b = tf.expand_dims(mu, axis=1)

    diff_matrix = tf.subtract(mu_b, mu_a)
    L_dist_1 = temp_norm(diff_matrix, axis=channel_axis)
    L_dist_2 = tf.square(tf.nn.relu(tf.constant(2 * delta_d, dtype=K.floatx()) - L_dist_1))
    diag = tf.constant(0, dtype=K.floatx()) * tf.diag_part(L_dist_2)
    L_dist_3 = tf.matrix_set_diag(L_dist_2, diag)
    L_dist = tf.reduce_mean(L_dist_3)

    # Compute regularization loss
    L_reg = gamma * temp_norm(mu, axis=-1)
    L = L_var + L_dist + tf.reduce_mean(L_reg)

    return L


def weighted_focal_loss(y_true, y_pred, n_classes=3, gamma=2., axis=None, from_logits=False):
    """Computes focal loss with class weights computed on the fly
    # Arguments:
        y_true: A tensor of the same shape as `y_pred`.
        y_pred: A tensor resulting from a softmax
    """
    if from_logits:
        raise Exception('weighted_focal_loss cannot take logits')
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else len(y_pred.get_shape()) - 1
    reduce_axis = [x for x in list(range(len(y_pred.get_shape()))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    y_true_cast = tf.cast(y_true, K.floatx())
    total_sum = tf.reduce_sum(y_true_cast)
    class_sum = tf.reduce_sum(y_true_cast, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / np.float(n_classes) * tf.divide(total_sum, class_sum + 1.)
    temp_loss = tf.multiply(tf.pow(1. - y_pred, gamma) * tf.log(y_pred), class_weights)
    focal_loss = - tf.reduce_sum(y_true * temp_loss, axis=axis)
    return focal_loss
