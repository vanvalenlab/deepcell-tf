"""
losses.py

Custom loss functions

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

def categorical_crossentropy(target, output, class_weights=None, axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
        result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else len(output.get_shape()) - 1
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis=axis, keep_dims=True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        if class_weights is None:
            return - tf.reduce_sum(target * tf.log(output), axis=axis)
        return - tf.reduce_sum(tf.multiply(target * tf.log(output), class_weights), axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)

def weighted_categorical_crossentropy(target, output, n_classes=3, axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    Automatically computes the class weights from the target image and uses
    them to weight the cross entropy
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
        result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if from_logits:
        raise Exception('weighted_categorical_crossentropy cannot take logits')
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else len(output.get_shape()) - 1
    reduce_axis = [x for x in list(range(len(output.get_shape()))) if x != axis]
    # scale preds so that the class probas of each sample sum to 1
    output /= tf.reduce_sum(output, axis=axis, keepdims=True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    target_cast = tf.cast(target, K.floatx())
    total_sum = tf.reduce_sum(target_cast)
    class_sum = tf.reduce_sum(target_cast, axis=reduce_axis, keepdims=True)
    class_weights = 1.0 / np.float(n_classes) * tf.divide(total_sum, class_sum + 1.)
    return - tf.reduce_sum(tf.multiply(target * tf.log(output), class_weights), axis=axis)

def sample_categorical_crossentropy(target, output, class_weights=None, axis=None, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor. Only the sampled
    pixels are used to compute the cross entropy
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
        (unless `from_logits` is True, in which
        case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
        result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if axis is None:
        axis = 1 if K.image_data_format() == 'channels_first' else len(output.get_shape()) - 1
    if not from_logits:
        # scale preds so that the class probabilities of each sample sum to 1
        output /= tf.reduce_sum(output, axis=axis, keep_dims=True)

        # Multiply with mask so that only the sampled pixels are used
        output = tf.multiply(output, target)

        # manual computation of crossentropy
        _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        if class_weights is None:
            return - tf.reduce_sum(target * tf.log(output), axis=axis)
        return - tf.reduce_sum(tf.multiply(target * tf.log(output), class_weights), axis=axis)
    return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred, smooth=1):
    return -dice_coef(y_true, y_pred, smooth)

def discriminative_instance_loss(y_true, y_pred, delta_v=0.5, delta_d=1.5, order=2, gamma=1e-3):

    def temp_norm(ten, axis=-1):
        return tf.sqrt(tf.constant(1e-4, dtype=K.floatx()) + tf.reduce_sum(tf.square(ten), axis=axis))

    # y_pred = tf.divide(y_pred, tf.expand_dims(tf.norm(y_pred, ord = 2, axis = -1), axis = -1))

    # Compute variance loss
    cells_summed = tf.tensordot(y_true, y_pred, axes=[[0, 1, 2], [0, 1, 2]])
    n_pixels = tf.cast(tf.count_nonzero(y_true, axis=[0, 1, 2]), dtype=K.floatx()) + K.epsilon()
    n_pixels_expand = tf.expand_dims(n_pixels, axis=1)
    mu = tf.divide(cells_summed, n_pixels_expand)

    mu_tensor = tf.tensordot(y_true, mu, axes=[[-1], [0]])
    L_var_1 = y_pred - mu_tensor
    L_var_2 = tf.square(tf.nn.relu(temp_norm(L_var_1, axis=-1) - tf.constant(delta_v, dtype=K.floatx())))
    L_var_3 = tf.tensordot(L_var_2, y_true, axes=[[0, 1, 2], [0, 1, 2]])
    L_var_4 = tf.divide(L_var_3, n_pixels)
    L_var = tf.reduce_mean(L_var_4)

    # Compute distance loss
    mu_a = tf.expand_dims(mu, axis=0)
    mu_b = tf.expand_dims(mu, axis=1)


    diff_matrix = tf.subtract(mu_a, mu_b)
    L_dist_1 = temp_norm(diff_matrix, axis=-1)
    L_dist_2 = tf.square(tf.nn.relu(tf.constant(2*delta_d, dtype=K.floatx()) - L_dist_1))
    diag = tf.constant(0, shape=[106], dtype=K.floatx())
    L_dist_3 = tf.matrix_set_diag(L_dist_2, diag)
    L_dist = tf.reduce_mean(L_dist_3)

    # Compute regularization loss
    L_reg = gamma * temp_norm(mu, axis=-1)

    L = L_var + L_dist + L_reg

    return L

def discriminative_instance_loss_3D(y_true, y_pred, delta_v=0.5, delta_d=1.5, order=2, gamma=1e-3):

    def temp_norm(ten, axis=-1):
        return tf.sqrt(tf.constant(1e-4, dtype=K.floatx()) + tf.reduce_sum(tf.square(ten), axis=axis))

    # y_pred = tf.divide(y_pred, tf.expand_dims(tf.norm(y_pred, ord = 2, axis = -1), axis = -1))

    # Compute variance loss
    cells_summed = tf.tensordot(y_true, y_pred, axes=[[0, 1, 2, 3], [0, 1, 2, 3]])
    n_pixels = tf.cast(tf.count_nonzero(y_true, axis=[0, 1, 2, 3]), dtype=K.floatx()) + K.epsilon()
    n_pixels_expand = tf.expand_dims(n_pixels, axis=1)
    mu = tf.divide(cells_summed, n_pixels_expand)

    mu_tensor = tf.tensordot(y_true, mu, axes=[[-1], [0]])
    L_var_1 = y_pred - mu_tensor
    L_var_2 = tf.square(tf.nn.relu(temp_norm(L_var_1, axis=-1) - tf.constant(delta_v, dtype=K.floatx())))
    L_var_3 = tf.tensordot(L_var_2, y_true, axes=[[0, 1, 2, 3], [0, 1, 2, 3]])
    L_var_4 = tf.divide(L_var_3, n_pixels)
    L_var = tf.reduce_mean(L_var_4)

    # Compute distance loss
    mu_a = tf.expand_dims(mu, axis=0)
    mu_b = tf.expand_dims(mu, axis=1)

    diff_matrix = tf.subtract(mu_a, mu_b)
    L_dist_1 = temp_norm(diff_matrix, axis=-1)
    L_dist_2 = tf.square(tf.nn.relu(tf.constant(2*delta_d, dtype=K.floatx()) - L_dist_1))
    diag = tf.constant(0, dtype=K.floatx()) * tf.diag_part(L_dist_2)
    L_dist_3 = tf.matrix_set_diag(L_dist_2, diag)
    L_dist = tf.reduce_mean(L_dist_3)

    # Compute regularization loss
    L_reg = gamma * temp_norm(mu, axis=-1)

    L = L_var + L_dist + L_reg

    return L
