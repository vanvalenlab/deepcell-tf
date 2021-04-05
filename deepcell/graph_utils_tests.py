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

import numpy as np


def _generate_test_images(img_w=21, img_h=21, num_channels=20):
    """ generates a multiplexed and single channel image with random pixel values

        Returns:
            list: index 0 is the multiplexed image and index 1 is the gray iamge
    """
    multiplexed_images = []
    gray_images = []
    for _ in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        imarray = np.random.rand(img_w, img_h, num_channels) * variance + bias
        multiplexed_images.append(imarray)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        gray_images.append(imarray)

    return [multiplexed_images, gray_images]

def max_cell_test(label_image):
    """ Compute the maximum number of cells in a single batch for a label image

    Args:
        label_image (int): Label image where every cell has been given a
            unique integer id

    Returns:
        np.int: The maximum number of cells in a single batch of the
            label image
    """


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


def get_cell_features_test(image, label_image):
    """Extract feature vectors from cells in multiplexed imaging data

    Args:
        image (float): Multiplexed imaging dataset
        label_image (int): Label image where every cell has been given a
            unique integer id

    Returns:
        np.array (float): Feature matrix consisting of feature vectors
            for each cell

    """


def get_celltypes_test(label_image, celltype_image):
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


def adj_to_degree_test(adj, power=-0.5, epsilon=1e-5):
    """ Convert adjacency matrix to degree matrix

    Args:
        adj: Adjacency matrix

    Returns:
        np.array: Degree matrix raised to a given power
    """


def normalize_adj_test(adj):
    """Symmetrically normalize adjacency matrix.

    Args:
        adj: Adjacency matrix

    Returns:
        np.array: Symmetrically normalized adjacency matrix in
            coordinate format
    """

def preprocess_adj_test(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to
    tuple representation.

    Args:
        adj: Adjacency matrix

    Returns:
        np.array: Symmetrically normalized adjacency matrix after adding self
            connections to each node
    """


def node_dist_test(prop0, prop1):
    """ Compute minimum distance between two cells

    Args:
        prop0: skimage regionprop object for cell 0
        prop1: skimage regionprop object for cell 1

    Returns:
        float: minimum distance between two cells
    """
