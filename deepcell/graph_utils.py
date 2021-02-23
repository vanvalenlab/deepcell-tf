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

import os
import cv2
import imageio

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow.keras.backend as K

from scipy.stats import mode
from scipy.spatial.distance import cdist

from skimage.io import imread
from skimage.external.tifffile import TiffFile
from skimage.morphology import binary_erosion, ball, disk
from skimage.measure import regionprops_table, regionprops

from sklearn.preprocessing import quantile_transform as qt


"""
Data Loading Functions
"""


def get_image(file_name):
    """Read image from file and returns it as a tensor

    Args:
        file_name (str): path to image file

    Returns:
        numpy.array: numpy array of image data
    """
    ext = os.path.splitext(file_name.lower())[-1]
    if ext in ['.tif', '.tiff']:
        return np.float32(TiffFile(file_name).asarray())
    return np.float32(imread(file_name))


def load_patient_data(path):
    """Load a single data set
    Args:
        path: Location of the dataset

    Returns:
        np.array: The multiplexed imaging dataset for a specific patient
    """

    channel_list = []

    for _, _, channels in os.walk(path):
        # Sort the channels so we know what order they are in
        channels.sort()

        for channel in channels:
            # Make sure we don't load the segmentations as features!
            if 'Segmentation' not in channel:
                channel_path = os.path.join(path, channel)
                img = get_image(channel_path)
                channel_list.append(img)
    patient_data = np.stack(channel_list, axis=-1)

    return patient_data


def load_mibi_data(path, point_list):
    """Load mibi dataset

    Args:
        path: Path containing the mibi dataset
        point_list: Which patients of the mibi dataset should be loaded
         * point_list = [2, 5, 8, 9, 21, 22, 24, 26, 34, 37, 38, 41]

    Returns:
        np.array: The mibi dataset for all of the points
    """

    # Get the full path for each point
    path_dict = {}
    for _, points, _ in os.walk(path):
        for point in points:
            patient_path = os.path.join(path, point)
            path_dict[point] = patient_path

    mibi_data = []
    
    for point in point_list:

        point = 'Point{}'.format(point)
        full_path = path_dict[point]
        patient_data = load_patient_data(full_path)

        # Make sure dimensions are 2048
        x_dim, y_dim, _ = patient_data.shape
        if (x_dim != 2048) or (y_dim != 2048):
            print(point)
        else:
            mibi_data.append(patient_data)

    return np.array(mibi_data)


def load_celltypes(path, point_list):
    """Load celltype dataset

    Args:
        path: Path containing the cell type dataset
        point_list: Which patients of the mibi/cell dataset should be loaded
         * point_list = [2, 5, 8, 9, 21, 22, 24, 26, 34, 37, 38, 41]

    Returns:
        np.array: The mibi dataset for all of the points
    """
    celltype_images = []
    for point in point_list:
        filename = "P{}_labeledImage.tiff".format(point)
        fullpath = os.path.join(path, filename)
        img = imageio.imread(fullpath)
        celltype_images.append(img)
    celltypes = np.stack(celltype_images, axis=0)
    celltypes = np.expand_dims(celltypes, axis=-1)
    return celltypes


"""
Functions for get cell information based on cell segmentations
* Feature Matrix, Adj Matrix, Cell type
"""


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

    # label image (batch, x, y, 1)
    # adjacency matrix (batch, Max cells, Max cells)

    # Remove singleton dimension
    label_image = np.squeeze(label_image)

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

        adjacency_matrix[batch,
                         0:distances.shape[0],
                         0:distances.shape[1]] = distances < distance_threshold

        if not self_connection:
            adjacency_matrix[batch,
                             0:distances.shape[0],
                             0:distances.shape[1]] -= np.eye(distances.shape[0])

        centroid_matrix[batch, 0:centroids.shape[0], :] = centroids
        label_matrix[batch, 0:labels.shape[0]] = labels

    return adjacency_matrix, centroid_matrix, label_matrix


def get_cell_features(image, label_image):
    """Extract feature vectors from cells in multiplexed imaging data

    Args:
        image (float): Multiplexed imaging dataset
        label_image (int): Label image where every cell has been given a
            unique integer id

    Returns:
        np.array (float): Feature matrix consisting of feature vectors
            for each cell

    """
    if image.shape[0] != label_image.shape[0]:
        raise ValueError("image and label_image must have the same size for "
                         "the batch dimmension")

    # Find out the max number of cells
    max_cells = get_max_cells(label_image)

    # Instantiate the feature matrix
    feature_matrix = np.zeros((image.shape[0], max_cells, image.shape[-1]))

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
            feature_matrix[batch, 0:mean_intensity.shape[0], feature] = mean_intensity

    return feature_matrix


def get_celltypes(label_image, celltype_image):
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

        try:
            props = regionprops_table(label_image_batch,
                                      properties=['coords', 'label'])


            for i in range(len(props['coords'])):
                coords = props['coords'][i]
                cell_type_list = celltype_image_batch[coords[:, 0], coords[:, 1]]
                cell_type = mode(cell_type_list, axis=None).mode[0]
                celltype_matrix[batch, i] = cell_type
        except:
            print(batch)

    return celltype_matrix


def get_cell_df(mibi_data, mibi_labels, mibi_celltypes, markers, marker_idx_dict):
    """Create a data frame that has all cell information in one place

    Args:
        mibi_data (np.array):
        mibi_labels (np.array):
        mibi_celltypes (np.array):
        markers (list):
        marker_idx_dict (dictionary):

    Returns:
        pd.DataFrame (float): cell by feature matrix with batch info and cell type

    """
    num_batches = mibi_data.shape[0]
    num_feats = mibi_data.shape[-1]

    cell_df = pd.DataFrame(0.0, index=range(0), columns=range(num_feats + 2))
    col_names = np.copy(markers).tolist()
    col_names.insert(0, 'label')
    col_names.insert(0, 'batch')
    cell_df.columns = col_names
    cell_df['cell_type'] = np.nan

    for batch in range(num_batches):
        mibi_image = mibi_data[batch]
        label_image = np.squeeze(mibi_labels[batch])
        type_image = np.squeeze(mibi_celltypes[batch])

        props = regionprops(label_image)

        num_cells = len(props)
        num_feats = mibi_data.shape[-1]

        batch_df = pd.DataFrame(0.0, index=range(num_cells), columns=range(num_feats + 2))
        batch_df.rename(columns={0:'batch', 1:'label'}, inplace = True)

        batch_df['batch'] = batch

        for feature in range(num_feats):
            marker = marker_idx_dict[feature]

            props_table = regionprops_table(label_image,
                                            intensity_image=mibi_image[..., feature],
                                            properties=['mean_intensity', 'label'])

            batch_df['label'] = props_table['label']
            batch_df[feature + 2] = props_table['mean_intensity']
            batch_df.rename(columns={feature+2: marker}, inplace=True)

        batch_df['cell_type'] = np.nan

        for prop in props:
            idx = batch_df[batch_df['label'] == prop.label].index[0]

            coords = prop.coords
            cts = type_image[coords[:, 0], coords[:, 1]]
            cell_type = mode(cts, axis=None).mode[0]

            batch_df.loc[idx, 'cell_type'] = cell_type

        cell_df = pd.concat([cell_df, batch_df])

    return cell_df


"""
Auxiliary functions
"""


def get_marker_dict(path):
    """Load a single dataset

    Args:
        path: Location of the dataset

    Returns:
        np.array: The multiplexed imaging dataset for a specific patient
    """

    marker_list = []

    for _, _, channels in os.walk(path):
        # Sort the channels so we know what order they are in
        channels.sort()
        for channel in channels:
            # Make sure we don't load the segmentations as features!
            if 'Segmentation' not in channel:
                marker_list.append(channel.split('.')[0])

    marker_dict = dict(zip(marker_list, list(range(len(marker_list)))))
    # marker_by_idx_dict = dict(zip(list(range(len(marker_list))),marker_list))

    return marker_dict


def adj_to_degree(adj, power=-0.5, epsilon=1e-5):
    """ Convert adjacency matrix to degree matrix

    Args:
        adj: Adjacency matrix

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
        adj: Adjacency matrix

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
        adj: Adjacency matrix

    Returns:
        np.array: Symmetrically normalized adjacency matrix after adding self
            connections to each node
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

    return adj_normalized


def cell_dist(prop0, prop1):
    """ Compute minimum distance between two cells

    Args:
        prop0: skimage regionprop object for cell 0
        prop1: skimage regionprop object for cell 1

    Returns:
        float: minimum distance between two cells
    """
    distance = cdist(prop0.coords, prop1.coords)
    min_distance = np.amin(distance)
    return min_distance


def erode_edges(mask, erosion_width):
    """Erode edge of objects to prevent them from touching

    Args:
        mask (numpy.array): uniquely labeled instance mask
        erosion_width (int): integer value for pixel width to erode edges

    Returns:
        numpy.array: mask where each instance has had the edges eroded

    Raises:
        ValueError: mask.ndim is not 2 or 3
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


def reshape_matrix(X, y, reshape_size=256):
    """
    Reshape matrix of dimension 4 to have x and y of size reshape_size.
    Adds overlapping slices to batches.
    E.g. reshape_size of 256 yields (1, 1024, 1024, 1) -> (16, 256, 256, 1)
    The input image is divided into subimages of side length reshape_size,
    with the last row and column of subimages overlapping the one before the last
    if the original image side lengths are not divisible by reshape_size.
    Args:
        X (numpy.array): raw 4D image tensor
        y (numpy.array): label mask of 4D image data
        reshape_size (int, list): size of the output tensor
            If input is int, output images are square with side length equal
            reshape_size. If it is a list of 2 ints, then the output images
            size is reshape_size[0] x reshape_size[1]
    Returns:
        numpy.array: reshaped X and y 4D tensors
                     in shape[1:3] = (reshape_size, reshape_size), if reshape_size is an int, and
                     shape[1:3] reshape_size, if reshape_size is a list of length 2
    Raises:
        ValueError: X.ndim is not 4
        ValueError: y.ndim is not 4
    """
    is_channels_first = K.image_data_format() == 'channels_first'
    if X.ndim != 4:
        raise ValueError('reshape_matrix expects X dim to be 4, got', X.ndim)
    if y.ndim != 4:
        raise ValueError('reshape_matrix expects y dim to be 4, got', y.ndim)

    if isinstance(reshape_size, int):
        reshape_size_x = reshape_size_y = reshape_size
    elif len(reshape_size) == 2 and all(isinstance(x, int) for x in reshape_size):
        reshape_size_x, reshape_size_y = reshape_size
    else:
        raise ValueError('reshape_size must be an integer or an iterable containing 2 integers.')

    image_size_x, image_size_y = X.shape[2:] if is_channels_first else X.shape[1:3]
    rep_number_x = np.int(np.ceil(np.float(image_size_x) / np.float(reshape_size_x)))
    rep_number_y = np.int(np.ceil(np.float(image_size_y) / np.float(reshape_size_y)))
    new_batch_size = X.shape[0] * rep_number_x * rep_number_y

    if is_channels_first:
        new_X_shape = (new_batch_size, X.shape[1], reshape_size_x, reshape_size_y)
        new_y_shape = (new_batch_size, y.shape[1], reshape_size_x, reshape_size_y)
    else:
        new_X_shape = (new_batch_size, reshape_size_x, reshape_size_y, X.shape[3])
        new_y_shape = (new_batch_size, reshape_size_x, reshape_size_y, y.shape[3])

    new_X = np.zeros(new_X_shape, dtype=K.floatx())
    new_y = np.zeros(new_y_shape, dtype='int32')

    counter = 0
    for b in range(X.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                _axis = 2 if is_channels_first else 1
                if i != rep_number_x - 1:
                    x_start, x_end = i * reshape_size_x, (i + 1) * reshape_size_x
                else:
                    x_start, x_end = -reshape_size_x, X.shape[_axis]

                if j != rep_number_y - 1:
                    y_start, y_end = j * reshape_size_y, (j + 1) * reshape_size_y
                else:
                    y_start, y_end = -reshape_size_y, y.shape[_axis + 1]

                if is_channels_first:
                    new_X[counter] = X[b, :, x_start:x_end, y_start:y_end]
                    new_y[counter] = y[b, :, x_start:x_end, y_start:y_end]
                else:
                    new_X[counter] = X[b, x_start:x_end, y_start:y_end, :]
                    new_y[counter] = y[b, x_start:x_end, y_start:y_end, :]

                new_y[counter] = relabel_movie(new_y[counter])
                counter += 1

    print('Reshaped feature data from {} to {}'.format(y.shape, new_y.shape))
    print('Reshaped training data from {} to {}'.format(X.shape, new_X.shape))
    return new_X, new_y


def relabel_movie(y):
    """Relabels unique instance IDs to be from 1 to N
    Args:
        y (numpy.array): tensor of integer labels
    Returns:
        numpy.array: relabeled tensor with sequential labels
    """
    new_y = np.zeros(y.shape)
    unique_cells = np.unique(y)  # get all unique values of y
    unique_cells = np.delete(unique_cells, 0)  # remove 0, as it is background
    relabel_ids = np.arange(1, len(unique_cells) + 1)
    for cell_id, relabel_id in zip(unique_cells, relabel_ids):
        cell_loc = np.where(y == cell_id)
        new_y[cell_loc] = relabel_id
    return new_y


def qt_transform(X):
    """Perform sklearn.preprocessing.quantile_transform on batch data

    Args:
        X: data to transform

    Returns:
        np.array: The mibi dataset transformed
    """

    batches = X.shape[-1]
    transformed_data = []

    for batch in range(batches):
        x_batch = X[..., batch]

        x_batch = qt(x_batch,
                     copy=False,
                     output_distribution='uniform',
                     n_quantiles=10)

        transformed_data.append(x_batch)

    return np.array(transformed_data)


def normalize_mibi_data(tmp_mibi_data, marker_idx_dict):
    """Normalize mibi data by applying a gaussian smothing on the raw data
       removing bottom 5% of labels and breaking remaining values into quantiles

    Args:
        tmp_mibi_data: raw mibi data loaded via load_mibi func
        marker_idx_dict: look up table to go from index to maker name

    Returns:
        np.array: normalized mibi_data
    """

    num_batches = tmp_mibi_data.shape[0]
    num_channels = tmp_mibi_data.shape[-1]

    mibi_data = []

    for batch in range(num_batches):
        mibi_batch = tmp_mibi_data[batch, ...]
        # channel_data = []

        channel_imgs = []
        channel_th = []
        for channel in range(num_channels):

            batch_channel_data = mibi_batch[..., channel]
            blur_img = cv2.GaussianBlur(batch_channel_data, (3, 3), 0)

            channel_imgs.append(blur_img)

            if len(blur_img[blur_img > 0]) == 0:
#                 print('batch {}'.format(batch), marker_idx_dict[channel])
                channel_th.append(0)
            else:
                low_vals = np.percentile(blur_img[blur_img > 0], 5)
                channel_th.append(low_vals)

        imgs = np.array(channel_imgs).T

        imgs.T[np.tile((np.sum((imgs < channel_th).T, axis=0) == num_channels),
                       (num_channels, 1)).reshape(imgs.T.shape)] = 0

        batch_data = qt_transform(imgs)

        mibi_data.append(batch_data.T)

    return np.array(mibi_data)


def celltype_to_labeled_img(mibi_celltypes, celltype_data):
    """
    DOC STRING
    """

    new_label_image = np.zeros(mibi_celltypes.shape)
    for batch in range(mibi_celltypes.shape[0]):
        print(batch)
        props = regionprops(mibi_celltypes[batch, ..., 0])
        for i, prop in enumerate(props):
            nli_batch = new_label_image[batch]
            nli_batch[prop.coords[:, 0], prop.coords[:, 1]] = celltype_data[batch, i]
        new_label_image[batch] = nli_batch


    return new_label_image

