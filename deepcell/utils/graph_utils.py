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
"""Utilities for graphs and GCNs"""

# import os

import numpy as np
import scipy.sparse as sp

from scipy.stats import mode
from scipy.spatial.distance import cdist

from skimage.io import imread
from skimage.measure import regionprops_table


def get_max_cells(label_image):
    """ Compute the maximum number of cells in a single batch for a label image

    Args:
        label_image (int): Label image where every cell has been given a
            unique integer id

    Returns:
        np.int: The maximum number of cells in a single batch of the
            label image
    """
    max_cells = 0
    for batch in range(label_image.shape[0]):
        new_max_cells = len(np.unique(label_image[batch])) - 1
        if new_max_cells > max_cells:
            max_cells = new_max_cells

    return max_cells


def image_to_graph(label_image,
                   distance_threshold=0.5,
                   self_connection=True):
    """Convert a label image to a graph

    Args:
        label_image (int): Label image where every cell has been given a
            unique integer id
        distance_threshold (float): Connect two cells with an edge if their
            centroids are closer than distance_threshold pixels
        self_connection (bool): keep or remove diagonal of adj matrix

    Returns:
        np.array: The adjacency matrix for the graph
    """
    # label image (batch, x, y, 1)
    # adjacency matrix (batch, Max cells, Max cells)

    # Remove singleton dimension (batch, num_pix_x, num_pix_y)
    label_image = np.squeeze(label_image, axis=-1)

    # Find out the maximum numbers of cells
    max_cells = get_max_cells(label_image)

    # Instantiate the adjacency matrix
    adjacency_matrix = np.zeros((label_image.shape[0], max_cells, max_cells))
    centroid_matrix = np.zeros((label_image.shape[0], max_cells, 2))
    label_matrix = np.zeros((label_image.shape[0], max_cells))

    for batch in range(label_image.shape[0]):
        label_image_batch = label_image[batch]

        props = regionprops_table(label_image_batch, properties=['centroid', 'label'])
        centroids = np.stack([props['centroid-0'], props['centroid-1']], axis=-1)
        labels = props['label']
        distances = cdist(centroids, centroids, metric='euclidean')

        threshold = distances.mean()*distance_threshold

        adjacency_matrix[batch,
                         0:distances.shape[0],
                         0:distances.shape[1]] = distances < threshold

        if not self_connection:
            adjacency_matrix[batch,
                             0:distances.shape[0],
                             0:distances.shape[1]] -= np.eye(distances.shape[0])

        centroid_matrix[batch, 0:centroids.shape[0], :] = centroids
        label_matrix[batch, 0:labels.shape[0]] = labels

    return adjacency_matrix, centroid_matrix, label_matrix


def get_cell_features(label_image, image):
    """Extract feature vectors from cells in multiplexed imaging data

    Args:
        image (float): Multiplexed imaging dataset
        label_image (int): Label image where every cell has been given a
            unique integer id

    Raises:
        ValueError: if image and label_image batch size does not match

    Returns:
        np.array: matrix consisting of feature vectors for each cell
    """
    if image.shape[0] != label_image.shape[0]:
        raise ValueError("image and label_image must have the same size for "
                         "the batch dimmension")

    # Find out the max number of cells
    max_cells = get_max_cells(label_image)

    # Instantiate the feature matrix
    feature_matrix = np.zeros((image.shape[0], max_cells, image.shape[-1]))
    label_matrix = np.zeros((label_image.shape[0], max_cells))

    # Extract the features
    # TODO: Add flag to include morphology features
    for batch in range(image.shape[0]):
        image_batch = image[batch]
        label_image_batch = np.squeeze(label_image[batch])
        for feature in range(image.shape[-1]):
            props = regionprops_table(label_image_batch,
                                      intensity_image=image_batch[..., feature],
                                      properties=['mean_intensity', 'label'])

            mean_intensity = props['mean_intensity']
            labels = props['label']

            feature_matrix[batch, 0:mean_intensity.shape[0], feature] = mean_intensity
            label_matrix[batch, 0:labels.shape[0]] = labels

    return feature_matrix


def get_celltypes(label_image, celltype_image):
    """Query cell type image with a given label image and get cell
    type for each cell

    Args:
        label_image (np.array): Label image where every cell has been given a
            unique integer id
        celltype_image (np.array): Image where pixels belonging to each cell
            are given an integer id specifying cell type

    Raises:
        ValueError: if image and label_image bach does not match

    Returns:
        np.array(float): Cell type matrix consisting of cell types for
            each cell
    """
    if label_image.shape[0] != celltype_image.shape[0]:
        raise ValueError("label_image and cell_type_image must have same size for "
                         " the batch dimension")

    # Find out max number of cells
    max_cells = get_max_cells(label_image)

    # Instantiate cell type matrix
    celltype_matrix = np.zeros((label_image.shape[0], max_cells))

    for batch in range(label_image.shape[0]):
        label_image_batch = np.squeeze(label_image[batch])
        celltype_image_batch = np.squeeze(celltype_image[batch])

        props = regionprops_table(label_image_batch,
                                  properties=['coords', 'label'])

        for i in range(len(props['coords'])):
            coords = props['coords'][i]
            cell_type_list = celltype_image_batch[coords[:, 0], coords[:, 1]]
            cell_type = mode(cell_type_list, axis=None).mode[0]
            celltype_matrix[batch, i] = cell_type

    return celltype_matrix


def adj_to_degree(adj, power=-0.5, epsilon=1e-5):
    """ Convert adjacency matrix to degree matrix

    Args:
        adj (float): Adjacency matrix
        power (float): tbd
        epsilon (float): factor to fix float error

    Returns:
        np.array: Degree matrix raised to a given power
    """
    
    # adj is (batch, row, col)
    degrees = np.sum(adj, axis=1)  # this should be (batch, col)

    degree_matrix = np.zeros(adj.shape)
    for batch, degree in enumerate(degrees):
        if power is not None:
            degree = (degree + epsilon) ** power
        degree_matrix[batch] = np.diagflat(degree)

    return degree_matrix


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.

    Args:
        adj (float): Adjacency matrix

    Returns:
        np.array: Symmetrically normalized adjacency matrix in
            coordinate format
    """
    d_mat_inv_sqrt = adj_to_degree(adj, power=-0.5)

    # Compute D^-0.5AD^0.5 - Recall that D and A are symmetric
    normalized_adjacency = np.matmul(d_mat_inv_sqrt, adj)
    normalized_adjacency = np.matmul(normalized_adjacency, d_mat_inv_sqrt)

    return normalized_adjacency


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to
    tuple representation.

    Args:
        adj (float): Adjacency matrix

    Returns:
        np.array: Symmetrically normalized adjacency matrix after adding self
            connections to each node
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

    return adj_normalized


def node_dist(prop0, prop1):
    """ Compute minimum distance between two cells

    Args:
        prop0 (object): skimage regionprop object for cell 0
        prop1 (object): skimage regionprop object for cell 1

    Returns:
        float: minimum distance between two cells
    """
    distance = cdist(prop0.coords, prop1.coords)
    min_distance = np.amin(distance)
    return min_distance
