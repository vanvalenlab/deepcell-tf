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
# ============================================================================
"""Utilities for data transformations
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import ndimage
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import ball, disk
from skimage.morphology import binary_erosion, binary_dilation
from tensorflow.python.keras import backend as K


def deepcell_transform(maskstack, dilation_radius=None, data_format=None):
    """
    Transforms a label mask for a z stack edge, interior, and background
    # Arguments:
        maskstack: label masks of uniquely labeled instances
        dilation_radius:  width to enlarge the edge feature of each instance
    # Returns:
        deepcell_stacks: masks of:
        [background_edge_feature, interior_edge_feature, interior_feature, background]
    """
    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = len(maskstack.shape) - 1

    maskstack = np.squeeze(maskstack, axis=channel_axis)

    # Detect the edges and interiors
    new_masks = np.zeros(maskstack.shape)
    edge_masks = np.zeros(maskstack.shape)
    strel = ball(1) if maskstack.ndim > 3 else disk(1)
    for cell_label in np.unique(maskstack):
        if cell_label != 0:
            for i in range(maskstack.shape[0]):
                # get the cell interior
                img = maskstack[i] == cell_label
                img = binary_erosion(img, strel)
                new_masks[i] += img

    interior_masks = np.multiply(new_masks, maskstack)
    edge_masks = (maskstack - interior_masks > 0).astype('int')
    interior_masks = (interior_masks > 0).astype('int')

    # dilate the background masks and subtract from all edges for background-edges
    dilated_background = np.zeros(maskstack.shape)
    for i in range(maskstack.shape[0]):
        background = (maskstack[i] == 0).astype('int')
        dilated_background[i] = binary_dilation(background, strel)

    background_edge_masks = (edge_masks - dilated_background > 0).astype('int')

    # edges that are not background-edges are interior-edges
    interior_edge_masks = (edge_masks - background_edge_masks > 0).astype('int')

    if dilation_radius:
        dil_strel = ball(dilation_radius) if maskstack.ndim > 3 else disk(dilation_radius)
        # Thicken cell edges to be more pronounced
        for i in range(edge_masks.shape[0]):
            interior_edge_masks[i] = binary_dilation(interior_edge_masks[i], selem=dil_strel)
            background_edge_masks[i] = binary_dilation(background_edge_masks[i], selem=dil_strel)

        # Thin the augmented edges by subtracting the interior features.
        interior_edge_masks = (interior_edge_masks - interior_masks > 0).astype('int')
        background_edge_masks = (background_edge_masks - interior_masks > 0).astype('int')

    background_masks = (1 - background_edge_masks - interior_edge_masks - interior_masks > 0)
    background_masks = background_masks.astype('int')

    all_stacks = [
        background_edge_masks,
        interior_edge_masks,
        interior_masks,
        background_masks
    ]

    deepcell_stacks = np.stack(all_stacks, axis=channel_axis)
    return deepcell_stacks


def erode_edges(mask, erosion_width):
    """Erode edge of objects to prevent them from touching
    # Arguments:
        mask: uniquely labeled instance mask
        erosion_width: integer value for pixel width to erode edges
    # Returns:
        mask where each instance has had the edges eroded
    """
    if erosion_width:
        new_mask = np.zeros(mask.shape)
        if mask.ndim == 2:
            strel = disk(erosion_width)
        elif mask.ndim == 3:
            strel = ball(erosion_width)
        else:
            raise ValueError('erode_edges expects arrays of ndim 2 or 3.'
                             'Got ndim: {}'.format(mask.ndim))
        for cell_label in np.unique(mask):
            if cell_label != 0:
                temp_img = mask == cell_label
                temp_img = binary_erosion(temp_img, strel)
                new_mask = np.where(mask == cell_label, temp_img, new_mask)
        return np.multiply(new_mask, mask).astype('int')
    return mask


def distance_transform_2d(mask, bins=16, erosion_width=None):
    """Transform a label mask into distance classes.
    # Arguments
        mask: a label mask (y data)
        bins: the number of transformed distance classes
        erosion_width: number of pixels to erode edges of each labels
    # Returns
        distance: a mask of same shape as input mask,
                  with each label being a distance class from 1 to bins
    """
    mask = np.squeeze(mask)  # squeeze the channels
    mask = erode_edges(mask, erosion_width)

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


def distance_transform_3d(maskstack, bins=4, erosion_width=None):
    """
    Transforms a label mask for a z stack into distance classes
    Uses scipy's distance_transform_edt
    # Arguments
        maskstack: a z-stack of label masks (y data)
        bins: the number of transformed distance classes
        erosion_width: number of pixels to erode edges of each labels
    # Returns
        distance: 3D Euclidiean Distance Transform
    """
    maskstack = np.squeeze(maskstack)  # squeeze the channels
    maskstack = erode_edges(maskstack, erosion_width)

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
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
             [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


def rotate_array_180(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
             [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]


def rotate_array_270(arr):
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
             [slice(None, None, -1), slice(None)]
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
