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
import pytest

import numpy as np

from deepcell import graph_utils

def _get_dummy_mibi_img_data(num_batches=3, img_dim=128):
    """dockersting holder - generate fake mibi data """
    # generate feature data
    
    multiplexed_img = []
    for _ in range(num_batches):
        bias = np.random.rand(img_dim, img_dim, 1)*64
        variance = np.random.rand(img_dim, img_dim, 1) * (255-64)
        imarray = np.random.rand(img_dim, img_dim, 6) * variance + bias
        multiplexed_img.append(imarray)
    
    multiplexed_img = np.array(multiplexed_img)

    return multiplexed_img.astype('float32')

def _get_dummy_mibi_label_data(num_batches=3, img_dim=128):
    """dockersting holder"""

    # generate label data
    labels = []
    num_cells = []
    while len(labels) < num_batches:
        _x = sk.data.binary_blobs(length=img_dim, n_dim=2)
        _y = sk.measure.label(_x)
        num_unique = len(np.unique(_y))
        if num_unique > 3:
            labels.append(_y)
            num_cells.append(num_unique)

    labels = np.stack(labels, axis=0)
    labels = np.expand_dims(labels, axis=-1)

    return labels.astype('int32'), max(num_cells)

def test_max_cell():
    """ Compute the maximum number of cells in a single batch for a label image

    Args:
        label_image (int): Label image where every cell has been given a
            unique integer id

    Returns:
        np.int: The maximum number of cells in a single batch of the
            label image
    """
    x = np.zeros((8, 8))

    x[0, 0] = 1
    x[7, 7] = 2
    x[7, 0] = 5
    x[4, 5] = 6
    x[0, 7] = 10

    x = np.expand_dims(x,axis=0)
    x = np.expand_dims(x,axis=-1)

    assert get_max_cells(x) == 5 
    


def test_image_to_graph(label_image,
                        distance_threshold=50,
                        self_connection=True):
    """Convert a label image to a graph

    Args:
        label_image (int): Label image where every cell has been given a
            unique integer id
        distance_threshold (float): Connect two cells with an edge if their
            centroids are closer than distance_threshold pixels

    Returns:
        np.array: The adjacency matrix for the graph
    """


def test_get_cell_features(image, label_image):
    """Extract feature vectors from cells in multiplexed imaging data

    Args:
        image (float): Multiplexed imaging dataset
        label_image (int): Label image where every cell has been given a
            unique integer id

    Returns:
        np.array (float): Feature matrix consisting of feature vectors
            for each cell

    """


def test_get_celltypes(label_image, celltype_image):
    """Query cell type image with a given label image and get cell
    type for each cell

    Args:
        label_image (np.array): Label image where every cell has been given a
            unique integer id
        cell_type_image (np.arry: Image where pixels belonging to each cell
            are given an integer id specifying cell type

    Returns:
        np.array (float): Cell type matrix consisting of cell types for
            each cell
    """

def test_adj_to_degree(adj, power=-0.5, epsilon=1e-5):
    """ Convert adjacency matrix to degree matrix

    Args:
        adj: Adjacency matrix

    Returns:
        np.array: Degree matrix raised to a given power
    """


def test_normalize_adj(adj):
    """Symmetrically normalize adjacency matrix.

    Args:
        adj: Adjacency matrix

    Returns:
        np.array: Symmetrically normalized adjacency matrix in
            coordinate format
    """

def test_preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to
    tuple representation.

    Args:
        adj: Adjacency matrix

    Returns:
        np.array: Symmetrically normalized adjacency matrix after adding self
            connections to each node
    """


def test_node_dist(prop0, prop1):
    """ Compute minimum distance between two cells

    Args:
        prop0: skimage regionprop object for cell 0
        prop1: skimage regionprop object for cell 1

    Returns:
        float: minimum distance between two cells
    """
