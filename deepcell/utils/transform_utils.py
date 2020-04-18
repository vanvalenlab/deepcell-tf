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
"""Utilities for data transformations"""

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

from deepcell_toolbox import erode_edges


def pixelwise_transform(mask, dilation_radius=None, data_format=None,
                        separate_edge_classes=False):
    """Transforms a label mask for a z stack edge, interior, and background

    Args:
        mask (tensor): tensor of labels
        dilation_radius (int):  width to enlarge the edge feature of
            each instance
        data_format (str): 'channels_first' or 'channels_last'
        separate_edge_classes (bool): Whether to separate the cell edge class
            into 2 distinct cell-cell edge and cell-background edge classes.

    Returns:
        numpy.array: one-hot encoded tensor of masks:
            if not separate_edge_classes: [cell_edge, cell_interior, background]
            otherwise: [bg_cell_edge, cell_cell_edge, cell_interior, background]
    """
    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 0
    else:
        channel_axis = -1

    # Detect the edges and interiors
    new_mask = np.zeros(mask.shape)
    strel = ball(1) if mask.ndim > 2 else disk(1)
    for cell_label in np.unique(mask):
        if cell_label != 0:
            # get the cell interior
            img = mask == cell_label
            img = binary_erosion(img, strel)
            new_mask += img

    interior = np.multiply(new_mask, mask)
    edge = (mask - interior > 0).astype('int')
    interior = (interior > 0).astype('int')

    if not separate_edge_classes:
        if dilation_radius:
            dil_strel = ball(dilation_radius) if mask.ndim > 2 else disk(dilation_radius)
            # Thicken cell edges to be more pronounced
            edge = binary_dilation(edge, selem=dil_strel)

            # Thin the augmented edges by subtracting the interior features.
            edge = (edge - interior > 0).astype('int')

        background = (1 - edge - interior > 0)
        background = background.astype('int')

        all_stacks = [
            edge,
            interior,
            background
        ]

        return np.stack(all_stacks, axis=channel_axis)

    # dilate the background masks and subtract from all edges for background-edges
    background = (mask == 0).astype('int')
    dilated_background = binary_dilation(background, strel)

    background_edge = (edge - dilated_background > 0).astype('int')

    # edges that are not background-edges are interior-edges
    interior_edge = (edge - background_edge > 0).astype('int')

    if dilation_radius:
        dil_strel = ball(dilation_radius) if mask.ndim > 2 else disk(dilation_radius)
        # Thicken cell edges to be more pronounced
        interior_edge = binary_dilation(interior_edge, selem=dil_strel)
        background_edge = binary_dilation(background_edge, selem=dil_strel)

        # Thin the augmented edges by subtracting the interior features.
        interior_edge = (interior_edge - interior > 0).astype('int')
        background_edge = (background_edge - interior > 0).astype('int')

    background = (1 - background_edge - interior_edge - interior > 0)
    background = background.astype('int')

    all_stacks = [
        background_edge,
        interior_edge,
        interior,
        background
    ]

    return np.stack(all_stacks, axis=channel_axis)


def distance_transform_2d(mask, bins=16, erosion_width=None):
    """Transform a label mask into distance classes.

    Args:
        mask (numpy.array): a label mask (y data)
        bins (int): the number of transformed distance classes
        erosion_width (int): number of pixels to erode edges of each labels

    Returns:
        numpy.array: a mask of same shape as input mask,
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


def distance_transform_continuous_2d(mask, erosion_width=None):
    """Transform a label mask into distance classes.

    Args:
        mask (numpy.array): a label mask (y data)
        bins (int): the number of transformed distance classes
        erosion_width (int): number of pixels to erode edges of each labels

    Returns:
        numpy.array: a mask of same shape as input mask,
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

    return distance  # minimum distance should be 0, not 1


def distance_transform_continuous_movie(mask, erosion_width=None):
    """Transform a label mask into a continuous distance value.

    Args:
        mask (numpy.array): a label mask (y data)
        erosion_width (int): number of pixels to erode edges of each labels

    Returns:
        numpy.array: a mask of same shape as input mask,
            with each label being a distance class from 1 to bins
    """
    distances = []
    for frame in range(mask.shape[0]):
        mask_frame = mask[frame]
        mask_frame = np.squeeze(mask_frame)  # squeeze the channels
        mask_frame = erode_edges(mask_frame, erosion_width)

        distance = ndimage.distance_transform_edt(mask_frame)
        distance = distance.astype(K.floatx())  # normalized distances are floats

        # uniquely label each cell and normalize the distance values
        # by that cells maximum distance value
        label_matrix = label(mask_frame)
        for prop in regionprops(label_matrix):
            labeled_distance = distance[label_matrix == prop.label]
            normalized_distance = labeled_distance / np.amax(labeled_distance)
            distance[label_matrix == prop.label] = normalized_distance
        distances.append(distance)

    distances = np.stack(distances, axis=0)

    return distances  # minimum distance should be 0, not 1


def centroid_transform_continuous_2d(mask, erosion_width=None, alpha=0.1):
    """Transform a label mask into a continuous centroid value.

    Args:
        mask (numpy.array): a label mask (y data)
        erosion_width (int): number of pixels to erode edges of each labels
        alpha (float, str): coefficent to reduce the magnitude of the distance
            value. If 'auto', determines alpha for each cell based on the cell
            area. Defaults to 0.1.

    Returns:
        numpy.array: a mask of same shape as input mask,
            with each label being a distance class from 1 to bins
    """

    # Check input to alpha
    if isinstance(alpha, str):
        if alpha.lower() != 'auto':
            raise ValueError('alpha must be set to "auto"')
            
    mask = np.squeeze(mask)
    mask = erode_edges(mask, erosion_width)

    distance = ndimage.distance_transform_edt(mask)
    distance = distance.astype(K.floatx())

    label_matrix = label(mask)

    inner_distance = np.zeros(distance.shape, dtype=K.floatx())
    for prop in regionprops(label_matrix, distance):
        coords = prop.coords
        center = prop.weighted_centroid
        distance_to_center = np.sum((coords - center) ** 2, axis=1)
        
        # Determine alpha to use
        if str(alpha).lower() == 'auto':
            _alpha = 1 / np.sqrt(prop.area)
        else:
            _alpha = float(alpha)

        center_transform = 1 / (1 + _alpha * distance_to_center)
        coords_x = coords[:, 0]
        coords_y = coords[:, 1]
        inner_distance[coords_x, coords_y] = center_transform

    return inner_distance


def centroid_transform_continuous_movie(mask, erosion_width=None, alpha=0.1):
    """Transform a label mask into a continuous centroid value.

    Args:
        mask (numpy.array): a label mask (y data)
        erosion_width (int): number of pixels to erode edges of each labels
        alpha (float): coefficent to reduce the magnitude of the distance
            value.

    Returns:
        numpy.array: a mask of same shape as input mask,
            with each label being a distance class from 1 to bins
    """
    inner_distances = []

    for frame in range(mask.shape[0]):
        mask_frame = mask[frame]
        mask_frame = np.squeeze(mask_frame)
        mask_frame = erode_edges(mask_frame, erosion_width)

        distance = ndimage.distance_transform_edt(mask_frame)
        distance = distance.astype(K.floatx())

        label_matrix = label(mask_frame)

        inner_distance = np.zeros(distance.shape, dtype=K.floatx())
        for prop in regionprops(label_matrix, distance):
            coords = prop.coords
            center = prop.weighted_centroid
            distance_to_center = np.sum((coords - center) ** 2, axis=1)
            center_transform = 1 / (1 + alpha * distance_to_center)
            coords_x = coords[:, 0]
            coords_y = coords[:, 1]
            inner_distance[coords_x, coords_y] = center_transform
        inner_distances.append(inner_distance)

    inner_distances = np.stack(inner_distances, axis=0)

    return inner_distances


def distance_transform_3d(maskstack, bins=4, erosion_width=None):
    """Transforms a label mask for a z stack into distance classes.
    Uses scipy's distance_transform_edt

    Args:
        maskstack (numpy.array): a z-stack of label masks (y data)
        bins (int): the number of transformed distance classes
        erosion_width (int): number of pixels to erode edges of each labels

    Returns:
        numpy.array: 3D Euclidiean Distance Transform
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


def centroid_weighted_distance_transform_2d(mask):
    """Transform a label mask into 2 distance masks weighted by the centroid.

    Args:
        mask (numpy.array): a label mask (y data)

    Returns:
        numpy.array: two masks of the same shape as input mask, with each label
            being a distance class scaled by the labels centroid
            (one image by the centroid's x-component and another by the y)
    """
    mask = mask.astype('int32')
    distance = ndimage.distance_transform_edt(mask)
    distance_x = ndimage.distance_transform_edt(mask)
    distance_y = ndimage.distance_transform_edt(mask)

    # normalized distances are floats
    distance_x = distance.astype(K.floatx())
    distance_y = distance.astype(K.floatx())

    # uniquely label each cell and normalize the distance values
    # by that cells maximum distance value before multiplying by
    # either the x-component of the centroid or y-component
    label_matrix = label(mask)
    for prop in regionprops(np.squeeze(label_matrix)):
        labeled_distance = distance[label_matrix == prop.label]
        normalized_distance = labeled_distance / np.amax(labeled_distance)
        y, x = prop.centroid
        distance_x[label_matrix == prop.label] = normalized_distance * x
        distance_y[label_matrix == prop.label] = normalized_distance * y
        # it may be better to use the following to cut down on
        # discrepancies in distance transform due to noise
        # distance_x[label_matrix == prop.label] = x
        # distance_y[label_matrix == prop.label] = y

    return distance_x, distance_y


def rotate_array_0(arr):
    """Rotate array 0 degrees

    Args:
        arr (numpy.array): input array

    Returns:
        numpy.array: rotated array
    """
    return arr


def rotate_array_90(arr):
    """Rotate array 90 degrees

    Args:
        arr (numpy.array): input array

    Returns:
        numpy.array: rotated array
    """
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
             [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)


def rotate_array_180(arr):
    """Rotate array 180 degrees

    Args:
        arr (numpy.array): input array

    Returns:
        numpy.array: rotated array
    """
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
             [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]


def rotate_array_270(arr):
    """Rotate array 270 degrees

    Args:
        arr (numpy.array): input array

    Returns:
        numpy.array: rotated array
    """
    axes_order = list(range(arr.ndim - 2)) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + \
             [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.

    Args:
        y (numpy.array): class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes (int): total number of classes.

    Returns:
        numpy.array: A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
