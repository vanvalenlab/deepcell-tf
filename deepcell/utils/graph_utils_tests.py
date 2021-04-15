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
   # create a test label image
    x = np.zeros((8, 8))
    x[0, 0] = 1
    x[7, 7] = 2
    x[7, 0] = 5
    x[4, 5] = 6
    x[0, 7] = 10
    
    # add a batch and a value dim (batch, img_dim, img_dim, 1)
    x = np.expand_dims(x,axis=0)
    x = np.expand_dims(x,axis=-1)

    assert graph_utils.get_max_cells(x) == 5


def test_image_to_graph():
    """Test the conversion a label image to a graph"""  

    # create a test label image
    x = np.zeros((8, 8))
    x[0, 0] = 1
    x[7, 7] = 2
    x[7, 0] = 5
    x[4, 5] = 6
    x[0, 7] = 10

    # add a batch and a value dim
    # x dims are (batch, img_dim, img_dim, 1)
    x = np.expand_dims(x,axis=0)
    x = np.expand_dims(x,axis=-1).astype('int')

    # define the true centriods based on x
    true_centroids = np.array([[0,0], [7,7], [7,0], [4,5], [0,7]])

    # define the true labels based on x
    true_labels = np.array([1,2,5,6,10])

    # test that NO connections are made when threshold = 0
    adj_mat, centriod_mat, label_mat = graph_utils.image_to_graph(x,
                                                                  distance_threshold=0.01,
                                                                  self_connection=True)
    assert np.sum(adj_mat) == 5
    assert (true_centroids == centriod_mat).all()
    assert (true_labels == label_mat).all()

    # test that there are 5 connections when threshold = 
    adj_mat, centriod_mat, label_mat = graph_utils.image_to_graph(x,
                                                                  distance_threshold=0.7,
                                                                  self_connection=True)
    assert np.sum(adj_mat) == 7.0
    assert (true_centroids == centriod_mat).all()
    assert (true_labels == label_mat).all()

    # test that graph is fully connected when threshold = 
    adj_mat, centriod_mat, label_mat = graph_utils.image_to_graph(x,
                                                                  distance_threshold=2.0,
                                                                  self_connection=True)
    assert np.sum(adj_mat) == 25.0
    assert (true_centroids == centriod_mat).all()
    assert (true_labels == label_mat).all()

    # test that NO connections are made when threshold = 0
    adj_mat, centriod_mat, label_mat = graph_utils.image_to_graph(x,
                                                                  distance_threshold=0.01,
                                                                  self_connection=False)
    assert np.sum(adj_mat) == 0
    assert (true_centroids == centriod_mat).all()
    assert (true_labels == label_mat).all()

    # test that there are 5 connections when threshold = 
    adj_mat, centriod_mat, label_mat = graph_utils.image_to_graph(x,
                                                                  distance_threshold=0.7,
                                                                  self_connection=False)
    assert np.sum(adj_mat) == 2.0
    assert (true_centroids == centriod_mat).all()
    assert (true_labels == label_mat).all()

    # test that graph is fully connected when threshold = 
    adj_mat, centriod_mat, label_mat = graph_utils.image_to_graph(x,
                                                                  distance_threshold=2.0,
                                                                  self_connection=False)
    assert np.sum(adj_mat) == 20.0
    assert (true_centroids == centriod_mat).all()
    assert (true_labels == label_mat).all()


def test_get_cell_features():
    """Test the extraction of feature vectors from cells in multiplexed imaging data"""

    # create a test label image
    x = np.zeros((8, 8))
    x[0, 0] = 1
    x[7, 7] = 2
    x[7, 0] = 5
    x[4, 5] = 6
    x[0, 7] = 10

    # add a batch and a value dim
    # x dims are (batch, img_dim, img_dim, 1)
    x = np.expand_dims(x,axis=0)
    x = np.expand_dims(x,axis=-1).astype('int')

    # create test image data
    y = [[2, 0, 1, 5, 3, 4, 0, 7],
         [6, 8, 3, 0, 7, 6, 2, 1],
         [6, 6, 5, 9, 5, 2, 2, 8],
         [5, 0, 1, 8, 6, 8, 5, 4],
         [7, 1, 0, 1, 1, 0, 0, 3],
         [5, 6, 9, 6, 0, 6, 6, 7],
         [8, 2, 0, 8, 4, 0, 5, 5],
         [8, 6, 3, 7, 2, 4, 6, 7]]

    # conver to numpy array
    y = np.array(y)

    # add a batch and a value dim
    # x dims are (batch, img_dim, img_dim, 1)
    y = np.expand_dims(y,axis=0)
    y = np.expand_dims(y,axis=-1)

    true_features = np.array([[2], [7], [8], [0], [7]])

    feature_matrix = graph_utils.get_cell_features(x, y)

    assert (true_features == feature_matrix).all()


def test_get_celltypes():
    """test the function for querying cell type image with a given label image and get cell
    type for each cell
    """
    # create a test label image
    x = np.zeros((8, 8))
    x[0, 0] = 1
    x[7, 7] = 2
    x[7, 0] = 5
    x[4, 5] = 6
    x[4, 4] = 6
    x[0, 7] = 10
    x[1, 7] = 10
    x[0, 6] = 10
    x[1, 6] = 10

    # add a batch and a value dim
    # x dims are (batch, img_dim, img_dim, 1)
    x = np.expand_dims(x,axis=0)
    x = np.expand_dims(x,axis=-1).astype('int')

    y = [[2, 2, 0, 2, 1, 1, 1, 1],
        [0, 2, 0, 0, 2, 1, 2, 1],
        [1, 2, 1, 1, 2, 1, 2, 2],
        [2, 1, 0, 1, 0, 0, 2, 2],
        [0, 1, 0, 0, 0, 0, 2, 0],
        [1, 0, 0, 0, 1, 0, 2, 1],
        [0, 1, 0, 2, 2, 1, 0, 2],
        [0, 0, 2, 2, 0, 1, 1, 2]]

    y = np.array(y)

    y = np.expand_dims(y,axis=0)
    y = np.expand_dims(y,axis=-1)

    # true cell types
    true_celltypes = np.array([2., 2., 0., 0., 1.])

    celltype_matrix = graph_utils.get_celltypes(x, y)

    assert (true_celltypes == celltype_matrix).all()

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
