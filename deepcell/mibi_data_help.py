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
# Functions for converting multiplex imaging dataset to graphs and delivering
# them as tensorflow dataset objects
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.stats import mode

from sklearn.model_selection import train_test_split

from skimage.measure import regionprops_table, regionprops

from tensorflow.keras.utils import to_categorical

from deepcell.utils.data_utils import reshape_matrix

from deepcell.graph_utils import get_celltypes
from deepcell.graph_utils import image_to_graph, get_cell_features, normalize_adj

from deepcell.tmp_funcs import normalize_mibi_data


def cells_dataset_from_df(cell_df_train, cell_df_val):
    """Docstring"""

    train_df = cell_df_train
    val_df = cell_df_val

    train_dataset = tf_dataset_from_df(train_df)
    val_dataset = tf_dataset_from_df(val_df)

    return train_dataset, train_df, val_dataset, val_df

def tf_dataset_from_df(dataframe):
    """docstring"""
    patient_ids = dataframe.pop('batch') # pylint: disable=W0612
    cell_labels = dataframe.pop('label') # pylint: disable=W0612
    targets = dataframe.pop('cell_type')

    dataset = tf.data.Dataset.from_tensor_slices(
        ({'feature_matrix': dataframe.values},
         {'celltypes': to_categorical(targets.values)}))
    return dataset


def data_split(multiplex_image,
               label_image,
               celltype_image,
               marker_idx_dict,
               train_val_cutoff=.85,
               normalize=True):
    """docstring"""

    training_cut = int(multiplex_image.shape[0]*train_val_cutoff)

    if normalize:
        mibi_train = normalize_mibi_data(multiplex_image[:training_cut,...], 
                                         marker_idx_dict)

        mibi_val = normalize_mibi_data(multiplex_image[training_cut:,...], 
                                       marker_idx_dict)
    else:
        mibi_train = multiplex_image[:training_cut,...]
        mibi_val = multiplex_image[training_cut:,...]

    training_data = [mibi_train,
                     label_image[:training_cut,...],
                     celltype_image[:training_cut,...]]

    validation_data = [mibi_val,
                       label_image[training_cut:,...],
                       celltype_image[training_cut:,...]]

    return training_data, validation_data


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

    # training_data, validation_data = data_split(mibi_data,
    #                                             mibi_labels,
    #                                             mibi_celltypes,
    #                                             marker_idx_dict)

    # mibi_data_train, mibi_labels_train, mibi_celltypes_train = training_data
    # mibi_data_val, mibi_labels_val, mibi_celltypes_val = validation_data

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

    idxs = np.unique(cell_df.batch)
    num_batches = len(idxs)
    test_size = num_batches - int(num_batches*.85)
    train_idxs, val_idxs = train_test_split(idxs, test_size=test_size)

    train_df = cell_df[cell_df.batch.isin(train_idxs)]
    val_df = cell_df[cell_df.batch.isin(val_idxs)]

    return train_df, val_df



def create_cell_tf_dataset(multiplex_image,
                           label_image,
                           celltype_image,
                           marker_idx_dict,
                           normalize=True,
                           mode='FC',
                           train_val_cutoff=.85,
                           reshape=False,
                           reshape_size=512):
    """docstring"""

    training_data, val_data = data_split(multiplex_image,
                                         label_image,
                                         celltype_image,
                                         marker_idx_dict,
                                         train_val_cutoff=train_val_cutoff,
                                         normalize=normalize)

    multiplex_train, label_train, celltype_train = training_data
    multiplex_val, label_val, celltype_val = val_data

    if reshape:
        multiplex_train, lb_train = reshape_matrix(multiplex_train, label_train, reshape_size)
        celltype_train, label_train = reshape_matrix(celltype_train, label_train, reshape_size)

        multiplex_val, lb_val = reshape_matrix(multiplex_val, label_val, reshape_size)
        celltype_val, label_val = reshape_matrix(celltype_val, label_val, reshape_size)



    train_adjacency_matrix, train_normalized_adjacency_matrix, train_coords = create_adjacency_matrix(label_train)
    train_feature_matrix = create_feature_matrix(multiplex_train, label_train)
    train_celltype_matrix = create_celltype_matrix(celltype_train, label_train)


    val_adjacency_matrix, val_normalized_adjacency_matrix, val_coords = create_adjacency_matrix(label_val)
    val_feature_matrix = create_feature_matrix(multiplex_val, label_val)
    val_celltype_matrix = create_celltype_matrix(celltype_val, label_val)


    if mode == 'FC':
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': train_feature_matrix},
             {'celltypes': train_celltype_matrix}))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': val_feature_matrix},
             {'celltypes': val_celltype_matrix}))

    elif mode == 'AE':
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': train_feature_matrix},
             {'decoded': train_feature_matrix}))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': val_feature_matrix},
             {'decoded': val_feature_matrix}))

    elif mode == 'AE_FC':
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': train_feature_matrix},
             {'decoded': train_feature_matrix,
              'celltypes': train_celltype_matrix}))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': val_feature_matrix},
             {'decoded': val_feature_matrix,
              'celltypes': val_celltype_matrix}))

    elif mode == 'GCN':
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': train_feature_matrix,
              'adjacency_matrix': train_normalized_adjacency_matrix},
             {'celltypes': train_celltype_matrix}))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': val_feature_matrix,
              'adjacency_matrix': val_normalized_adjacency_matrix},
             {'celltypes': val_celltype_matrix}))         

    elif mode == 'GAE':
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': train_feature_matrix,
              'adjacency_matrix': train_normalized_adjacency_matrix},
             {'decoded': train_adjacency_matrix}))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': val_feature_matrix,
              'adjacency_matrix': val_normalized_adjacency_matrix},
             {'decoded': val_adjacency_matrix}))        

    elif mode == 'GAE_FC':
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': train_feature_matrix,
              'adjacency_matrix': train_normalized_adjacency_matrix},
             {'decoded': train_adjacency_matrix,
              'celltypes': train_celltype_matrix}))

        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'feature_matrix': val_feature_matrix,
              'adjacency_matrix': val_normalized_adjacency_matrix},
             {'decoded': val_adjacency_matrix,
              'celltypes': val_celltype_matrix}))

    elif mode == 'EGCL':
        train_dataset = tf.data.Dataset.from_tensor_slices(
          ({'feature_matrix': train_feature_matrix,
            'adjacency_matrix': train_normalized_adjacency_matrix,
            'coords': train_coords,},
           {'celltypes': train_celltype_matrix}))

        val_dataset = tf.data.Dataset.from_tensor_slices(
          ({'feature_matrix': val_feature_matrix,
            'adjacency_matrix': val_normalized_adjacency_matrix,
            'coords': val_coords,},
           {'celltypes': val_celltype_matrix}))        

    else:
        print('please enter a valid mode type')
        print()
        print('FC, AE, AE_FC, GAE, GAE_FC')

    return train_dataset, val_dataset


def create_adjacency_matrix(label_image):
    """docstring"""
    adjacency_matrix, centroid_matrix, _ = image_to_graph(label_image,
                                                                     self_connection=True)
    adjacency_matrix.astype(np.float32)
    normalized_adjacency_matrix = normalize_adj(adjacency_matrix)

    return adjacency_matrix, normalized_adjacency_matrix, centroid_matrix

def create_feature_matrix(multiplex_image, label_image):
    """docstring"""
    feature_matrix = get_cell_features(multiplex_image,
                                       label_image).astype(np.float32)

    return feature_matrix


def create_celltype_matrix(celltype_image, label_image):
    """docstring"""
    celltype_data = get_celltypes(label_image,
                                  celltype_image)

    cell_type_data = to_categorical(celltype_data)

    return cell_type_data
