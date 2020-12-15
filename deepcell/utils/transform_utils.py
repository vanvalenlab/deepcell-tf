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
"""Utilities for data transformations"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import ndimage
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import ball, disk
from skimage.morphology import binary_erosion
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries
from tensorflow.keras import backend as K

from deepcell_toolbox import erode_edges


def pixelwise_transform(mask, dilation_radius=None, data_format=None,
                        separate_edge_classes=False):
    """Transforms a label mask for a z stack edge, interior, and background

    Args:
        mask (numpy.array): tensor of labels
        dilation_radius (int):  width to enlarge the edge feature of
            each instance
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        separate_edge_classes (bool): Whether to separate the cell edge class
            into 2 distinct cell-cell edge and cell-background edge classes.

    Returns:
        numpy.array: An array with the same shape as ``mask``, except the
        channel axis will be a one-hot encoded semantic segmentation for
        3 main features:
        ``[cell_edge, cell_interior, background]``.
        If ``separate_edge_classes`` is ``True``, the ``cell_interior``
        feature is split into 2 features and the resulting channels are:
        ``[bg_cell_edge, cell_cell_edge, cell_interior, background]``.
    """
    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 0
    else:
        channel_axis = -1

    # Detect the edges and interiors
    edge = find_boundaries(mask, mode='inner').astype('int')
    interior = np.logical_and(edge == 0, mask > 0).astype('int')

    strel = ball(1) if mask.ndim > 2 else disk(1)
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


def outer_distance_transform_2d(mask, bins=None, erosion_width=None,
                                normalize=True):
    """Transform a label mask with an outer distance transform.

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes. If ``None``,
            returns the continuous outer transform.
        erosion_width (int): Number of pixels to erode edges of each labels
        normalize (bool): Normalize the transform of each cell by that
            cell's largest distance.

    Returns:
        numpy.array: A mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.
    """
    mask = np.squeeze(mask)  # squeeze the channels
    mask = erode_edges(mask, erosion_width)

    distance = ndimage.distance_transform_edt(mask)
    distance = distance.astype(K.floatx())  # normalized distances are floats

    if normalize:
        # uniquely label each cell and normalize the distance values
        # by that cells maximum distance value
        label_matrix = label(mask)
        for prop in regionprops(label_matrix):
            labeled_distance = distance[label_matrix == prop.label]
            normalized_distance = labeled_distance / np.amax(labeled_distance)
            distance[label_matrix == prop.label] = normalized_distance

    if bins is None:
        return distance

    # bin each distance value into a class from 1 to bins
    min_dist = np.amin(distance)
    max_dist = np.amax(distance)
    distance_bins = np.linspace(min_dist - K.epsilon(),
                                max_dist + K.epsilon(),
                                num=bins + 1)
    distance = np.digitize(distance, distance_bins, right=True)
    return distance - 1  # minimum distance should be 0, not 1


def outer_distance_transform_3d(mask, bins=None, erosion_width=None,
                                normalize=True, sampling=[0.5, 0.217, 0.217]):
    """Transforms a label mask for a z stack with an outer distance transform.
    Uses scipy's distance_transform_edt

    Args:
        mask (numpy.array): A z-stack of label masks (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): Number of pixels to erode edges of each labels.
        normalize (bool): Normalize the transform of each cell by that
            cell's largest distance.
        sampling (list): Spacing of pixels along each dimension.

    Returns:
        numpy.array: 3D Euclidiean Distance Transform
    """
    maskstack = np.squeeze(mask)  # squeeze the channels
    maskstack = erode_edges(maskstack, erosion_width)

    distance = ndimage.distance_transform_edt(maskstack, sampling=sampling)

    # normalize by maximum distance
    if normalize:
        for cell_label in np.unique(maskstack):
            if cell_label == 0:  # distance is only found for non-zero regions
                continue
            index = np.nonzero(maskstack == cell_label)
            distance[index] = distance[index] / np.amax(distance[index])

    if bins is None:
        return distance

    # divide into bins
    min_dist = np.amin(distance.flatten())
    max_dist = np.amax(distance.flatten())
    distance_bins = np.linspace(min_dist - K.epsilon(),
                                max_dist + K.epsilon(),
                                num=bins + 1)
    distance = np.digitize(distance, distance_bins, right=True)
    return distance - 1  # minimum distance should be 0, not 1


def outer_distance_transform_movie(mask, bins=None, erosion_width=None,
                                   normalize=True):
    """Transform a label mask for a movie with an outer distance transform.
    Applies the 2D transform to each frame.

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): number of pixels to erode edges of each labels.
        normalize (bool): Normalize the transform of each cell by that
            cell's largest distance.

    Returns:
        numpy.array: a mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``
    """
    distances = []
    for frame in range(mask.shape[0]):
        mask_frame = mask[frame]

        distance = outer_distance_transform_2d(
            mask_frame, bins=bins,
            erosion_width=erosion_width,
            normalize=normalize)

        distances.append(distance)

    distances = np.stack(distances, axis=0)

    return distances


def inner_distance_transform_2d(mask, bins=None, erosion_width=None,
                                alpha=0.1, beta=1):
    """Transform a label mask with an inner distance transform.

    .. code-block:: python

        inner_distance = 1 / (1 + beta * alpha * distance_to_center)

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): number of pixels to erode edges of each labels
        alpha (float, str): coefficent to reduce the magnitude of the distance
            value. If "auto", determines ``alpha`` for each cell based on the
            cell area.
        beta (float): scale parameter that is used when ``alpha`` is "auto".

    Returns:
        numpy.array: a mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.

    Raises:
        ValueError: ``alpha`` is a string but not set to "auto".
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

        center_transform = 1 / (1 + beta * _alpha * distance_to_center)
        coords_x = coords[:, 0]
        coords_y = coords[:, 1]
        inner_distance[coords_x, coords_y] = center_transform

    if bins is None:
        return inner_distance

    # divide into bins
    min_dist = np.amin(inner_distance.flatten())
    max_dist = np.amax(inner_distance.flatten())
    distance_bins = np.linspace(min_dist - K.epsilon(),
                                max_dist + K.epsilon(),
                                num=bins + 1)
    inner_distance = np.digitize(inner_distance, distance_bins, right=True)
    return inner_distance - 1  # minimum distance should be 0, not 1


def inner_distance_transform_3d(mask, bins=None,
                                erosion_width=None,
                                alpha=0.1, beta=1,
                                sampling=[0.5, 0.217, 0.217]):
    """Transform a label mask for a z-stack with an inner distance transform.

    .. code-block:: python

        inner_distance = 1 / (1 + beta * alpha * distance_to_center)

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): Number of pixels to erode edges of each labels
        alpha (float, str): Coefficent to reduce the magnitude of the distance
            value. If ``'auto'``, determines alpha for each cell based on the
            cell area.
        beta (float): Scale parameter that is used when ``alpha`` is "auto".
        sampling (list): Spacing of pixels along each dimension.

    Returns:
        numpy.array: A mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.

    Raises:
        ValueError: ``alpha`` is a string but not set to "auto".
    """
    # Check input to alpha
    if isinstance(alpha, str):
        if alpha.lower() != 'auto':
            raise ValueError('alpha must be set to "auto"')

    mask = np.squeeze(mask)
    mask = erode_edges(mask, erosion_width)

    distance = ndimage.distance_transform_edt(mask, sampling=sampling)
    distance = distance.astype(K.floatx())

    label_matrix = label(mask)

    inner_distance = np.zeros(distance.shape, dtype=K.floatx())
    for prop in regionprops(label_matrix, distance):
        coords = prop.coords
        center = prop.weighted_centroid
        distance_to_center = (coords - center) * np.array(sampling)
        distance_to_center = np.sum(distance_to_center ** 2, axis=1)

        # Determine alpha to use
        if str(alpha).lower() == 'auto':
            _alpha = 1 / np.cbrt(prop.area)
        else:
            _alpha = float(alpha)

        center_transform = 1 / (1 + beta * _alpha * distance_to_center)
        coords_z = coords[:, 0]
        coords_x = coords[:, 1]
        coords_y = coords[:, 2]
        inner_distance[coords_z, coords_x, coords_y] = center_transform

    if bins is None:
        return inner_distance

    # divide into bins
    min_dist = np.amin(inner_distance.flatten())
    max_dist = np.amax(inner_distance.flatten())
    distance_bins = np.linspace(min_dist - K.epsilon(),
                                max_dist + K.epsilon(),
                                num=bins + 1)
    inner_distance = np.digitize(inner_distance, distance_bins, right=True)
    return inner_distance - 1  # minimum distance should be 0, not 1


def inner_distance_transform_movie(mask, bins=None, erosion_width=None,
                                   alpha=0.1, beta=1):
    """Transform a label mask with an inner distance transform. Applies the
    2D transform to each frame.

    Args:
        mask (numpy.array): A label mask (``y`` data).
        bins (int): The number of transformed distance classes.
        erosion_width (int): Number of pixels to erode edges of each labels.
        alpha (float, str): Coefficent to reduce the magnitude of the distance
            value. If "auto", determines ``alpha`` for each cell based on the
            cell area.
        beta (float): Scale parameter that is used when ``alpha`` is "auto".

    Returns:
        numpy.array: A mask of same shape as input mask,
        with each label being a distance class from 1 to ``bins``.

    Raises:
        ValueError: ``alpha`` is a string but not set to "auto".
    """
    # Check input to alpha
    if isinstance(alpha, str):
        if alpha.lower() != 'auto':
            raise ValueError('alpha must be set to "auto"')

    inner_distances = []

    for frame in range(mask.shape[0]):
        mask_frame = mask[frame]

        inner_distance = inner_distance_transform_2d(
            mask_frame, bins=bins,
            erosion_width=erosion_width,
            alpha=alpha, beta=beta)

        inner_distances.append(inner_distance)

    inner_distances = np.stack(inner_distances, axis=0)

    return inner_distances
