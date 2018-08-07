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
    bins = np.linspace(min_dist - K.epsilon(), max_dist + K.epsilon(), num=bins)
    distance = np.digitize(distance, bins)
    return distance

def centroid_weighted_distance_transform_2d(mask):
    """Transform a label mask into 2 scaled distance masks weighted by the element's centroid.
    # Arguments
        mask: a label mask (y data)
    # Returns
        distance: two masks of the same shape as input mask,
                  with each label being a distance class scaled by the labels centroid
                  (one image by the x-component of the centroid and another by the y)
    """
    distance = ndimage.distance_transform_edt(np.int32(mask))
    distance_x = ndimage.distance_transform_edt(np.int32(mask))
    distance_y = ndimage.distance_transform_edt(np.int32(mask))
    
    distance_x = distance.astype(K.floatx())  # normalized distances are floats
    distance_y = distance.astype(K.floatx())

    # uniquely label each cell and normalize the distance values
    # by that cells maximum distance value before multiplying by 
    # either the x-component of the centroid or y-component
    label_matrix = label(np.int32(mask))
    for prop in regionprops(label_matrix):
        labeled_distance = distance[label_matrix == prop.label]
        normalized_distance = labeled_distance / np.amax(labeled_distance)
        y, x = prop.centroid
        distance_x[label_matrix == prop.label] = normalized_distance * x
        distance_y[label_matrix == prop.label] = normalized_distance * y
        #may be better to use the following to cut down on discrepancies in distance 
        #transform due to noise
        #distance_x[label_matrix == prop.label] = x
        #distance_y[label_matrix == prop.label] = y
 
    return distance_x, distance_y

def rotate_array_0(arr):
    return arr

def rotate_array_90(arr):
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim-1, arr.ndim-2]
    slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)

def rotate_array_180(arr):
    slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]

def rotate_array_270(arr):
    axes_order = list(range(arr.ndim-2)) + [arr.ndim-1, arr.ndim-2]
    slices = [slice(None) for _ in range(arr.ndim-2)] + [slice(None, None, -1), slice(None)]
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

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
