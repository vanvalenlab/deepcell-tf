"""
transform_utils.py

Functions for data transformations

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import ndimage
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import ball
from skimage.morphology import binary_erosion
from tensorflow.python.keras import backend as K


def distance_transform_2d(mask, bins=16):
    """Transform a label mask into distance classes.
    # Arguments
        mask: a label mask (y data)
        bins: the number of transformed distance classes
    # Returns
        distance: a mask of same shape as input mask,
                  with each label being a distance class from 1 to bins
    """
    mask = np.squeeze(mask)  # squeeze the channels
    distance = ndimage.distance_transform_edt(mask)
    distance = distance.astype(K.floatx())  # normalized distances are floats

    # uniquely label each cell and normalize the distance values
    # by that cells maximum distance value
    label_matrix = label(mask)
    for prop in regionprops(label_matrix):
        labeled_distance = distance[label_matrix == prop.label]
        normalized_distance = labeled_distance / np.amax(labeled_distance)
        distance[label_matrix == prop.label] = normalized_distance

    # bin each distance value into a class from 1 to bins
    min_dist = np.amin(distance)
    max_dist = np.amax(distance)
    bins = np.linspace(min_dist - K.epsilon(), max_dist + K.epsilon(), num=bins + 1)
    distance = np.digitize(distance, bins, right=True)
    return distance - 1  # minimum distance should be 0, not 1


def _distance_transform_3d(maskstack, bins=16):
    """ # DEPRECATED
    Transform a label mask into distance classes for a z-stack of images
    # Arguments
        mask: a z-stack of label masks (y data)
        bins: the number of transformed distance classes
    # Returns
        distance: 3D Euclidiean Distance Transform
    """
    def weightmask(mask):
        # if mask is binary create unique labels
        img = label(mask) if np.unique(mask).size <= 2 else mask
        img = mask.flatten()
        unique, counts = np.unique(img, return_counts=True)
        counts = 1 / np.sqrt(counts)
        dic = dict(zip(unique, counts))
        dic[0] = 0
        img_out = map(dic.get, img)
        img_out = list(img_out)
        new_shape = (mask.shape[0], mask.shape[1], mask.shape[2])
        return np.reshape(img_out, new_shape)

    weighted_mask = weightmask(maskstack)
    distance_slices = [ndimage.distance_transform_edt(m) for m in weighted_mask]
    distance_slices = np.array(distance_slices)

    distance = np.zeros(list(weighted_mask.shape))
    for k in range(weighted_mask.shape[0]):
        adder = [np.square(x - k) for x in range(len(distance_slices))]
        for i in range(weighted_mask.shape[1]):
            for j in range(weighted_mask.shape[2]):
                slicearr = np.square(distance_slices[:, i, j])
                zans = np.argmin(slicearr + adder)
                zij = np.square(distance_slices[zans, i, j])
                zk = np.square(zans - k)
                distance[k][i][j] = np.sqrt(zij + zk)

    # normalize by maximum distance
    distance = np.expand_dims(distance, axis=-1)  # add channels for comparison
    for cell_label in np.unique(maskstack):
        if cell_label == 0:  # distance is only found for non-zero regions
            continue
        labeled_distance = distance[maskstack == cell_label]
        normalized_distance = labeled_distance / np.amax(labeled_distance)
        distance[maskstack == cell_label] = normalized_distance
    distance = np.reshape(distance, distance.shape[:-1])  # remove channels again

    min_dist = np.amin(distance.flatten())
    max_dist = np.amax(distance.flatten())
    bins = np.linspace(min_dist - K.epsilon(), max_dist + K.epsilon(), num=bins + 1)
    distance = np.digitize(distance, bins, right=True)
    return distance - 1  # minimum distance should be 0, not 1


def distance_transform_3d(maskstack, bins=4, erosion_width=None):
    """
    Transforms a label mask for a z stack into distance classes
    Uses scipy's distance_transform_edt
    # Arguments
        maskstack: a z-stack of label masks (y data)
        bins: the number of transformed distance classes
        erode_edges: erode edges of labels to prevent them from touching
    # Returns
        distance: 3D Euclidiean Distance Transform
    """
    maskstack = np.squeeze(maskstack)  # squeeze the channels
    # Erode masks
    new_masks = np.zeros(maskstack.shape)

    if erosion_width:
        strel = ball(erosion_width)
        for cell_label in np.unique(maskstack):
            if cell_label != 0:
                temp_img = maskstack == cell_label
                temp_img = binary_erosion(temp_img, strel)
                new_masks = np.where(maskstack == cell_label, temp_img, maskstack)

        maskstack = np.multiply(new_masks, maskstack)

    distance = ndimage.distance_transform_edt(maskstack, sampling=[0.5, 0.217, 0.217])

    # normalize by maximum distance
    for cell_label in np.unique(maskstack):
        if cell_label == 0:  # distance is only found for non-zero regions
            continue
        index = np.nonzero(maskstack == cell_label)
        distance[index] = distance[index] / np.amax(distance[index])
    # divide into bins
    min_dist = np.amin(distance.flatten())
    max_dist = np.amax(distance.flatten())
    bins = np.linspace(min_dist - K.epsilon(), max_dist + K.epsilon(), num=bins + 1)
    distance = np.digitize(distance, bins, right=True)
    return distance - 1  # minimum distance should be 0, not 1


def rotate_array_0(arr):
    return arr


def rotate_array_90(arr):
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


def rotate_array_180(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]


def rotate_array_270(arr):
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([
        [1, 0, o_x],
        [0, 1, o_y],
        [0, 0, 1]
    ])
    reset_matrix = np.array([
        [1, 0, -o_x],
        [0, 1, -o_y],
        [0, 0, 1]
    ])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
