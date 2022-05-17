# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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
"""Dataset Builders"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from scipy.spatial.distance import cdist

import numpy as np
import tqdm

import tensorflow as tf
import tensorflow_addons as tfa

from deepcell_tracking.utils import load_trks
from deepcell_tracking.utils import get_max_cells
from deepcell_tracking.utils import get_image_features
from deepcell_tracking.utils import is_valid_lineage
from deepcell_tracking.utils import normalize_adj_matrix
from deepcell_tracking.utils import relabel_sequential_lineage

from deepcell.data import split_dataset


class Track(object):  # pylint: disable=useless-object-inheritance

    def __init__(self, path=None, tracked_data=None,
                 appearance_dim=32, distance_threshold=64):
        if tracked_data:
            training_data = tracked_data
        elif path:
            training_data = load_trks(path)
        else:
            raise ValueError('One of `tracked_data` or `path` is required')
        self.X = training_data['X'].astype('float32')
        self.y = training_data['y'].astype('int32')
        self.lineages = training_data['lineages']
        if not len(self.X) == len(self.y) == len(self.lineages):
            raise ValueError(
                'The data do not share the same batch size. '
                'Please make sure you are using a valid .trks file')
        self.appearance_dim = appearance_dim
        self.distance_threshold = distance_threshold

        # Correct lineages and remove bad batches
        self._correct_lineages()

        # Create feature dictionaries
        features_dict = self._get_features()

        self.appearances = features_dict['appearances']
        self.morphologies = features_dict['morphologies']
        self.centroids = features_dict['centroids']

        # Convert adj matrices to sparse
        self.adj_matrices = self._get_sparse(
            features_dict['adj_matrix'])
        self.norm_adj_matrices = self._get_sparse(
            normalize_adj_matrix(features_dict['adj_matrix']))
        self.temporal_adj_matrices = self._get_sparse(
            features_dict['temporal_adj_matrix'])

        self.mask = features_dict['mask']
        self.track_length = features_dict['track_length']

    def _correct_lineages(self):
        """Ensure valid lineages and sequential labels for all batches"""
        new_X = []
        new_y = []
        new_lineages = []
        for batch in tqdm.tqdm(range(self.y.shape[0])):
            if is_valid_lineage(self.y[batch], self.lineages[batch]):

                y_relabel, new_lineage = relabel_sequential_lineage(
                    self.y[batch], self.lineages[batch])

                new_X.append(self.X[batch])
                new_y.append(y_relabel)
                new_lineages.append(new_lineage)

        self.X = np.stack(new_X, axis=0)
        self.y = np.stack(new_y, axis=0)
        self.lineages = new_lineages

    def _get_sparse(self, arr):
        # tf.sparse.from_dense causes memory issues for large arrays
        indices = np.nonzero(arr)
        indices_array = np.stack(indices, axis=-1)
        values = arr[indices]
        shape = arr.shape
        return tf.sparse.SparseTensor(indices_array, values, shape)

    def _get_features(self):
        """
        Extract the relevant features from the label movie
        Appearance, morphologies, centroids, and adjacency matrices
        """
        max_tracks = get_max_cells(self.y)
        n_batches = self.X.shape[0]
        n_frames = self.X.shape[1]
        n_channels = self.X.shape[-1]

        batch_shape = (n_batches, n_frames, max_tracks)

        appearance_shape = (self.appearance_dim, self.appearance_dim, n_channels)

        appearances = np.zeros(batch_shape + appearance_shape, dtype='float32')

        morphologies = np.zeros(batch_shape + (3,), dtype='float32')

        centroids = np.zeros(batch_shape + (2,), dtype='float32')

        adj_matrix = np.zeros(batch_shape + (max_tracks,), dtype='float32')

        temporal_adj_matrix = np.zeros((n_batches,
                                        n_frames - 1,
                                        max_tracks,
                                        max_tracks,
                                        3), dtype='float32')

        mask = np.zeros(batch_shape, dtype='float32')

        track_length = np.zeros((n_batches, max_tracks, 2), dtype='int32')

        for batch in tqdm.tqdm(range(n_batches)):
            for frame in range(n_frames):

                frame_features = get_image_features(
                    self.X[batch, frame], self.y[batch, frame],
                    appearance_dim=self.appearance_dim)

                track_ids = frame_features['labels'] - 1
                centroids[batch, frame, track_ids] = frame_features['centroids']
                morphologies[batch, frame, track_ids] = frame_features['morphologies']
                appearances[batch, frame, track_ids] = frame_features['appearances']
                mask[batch, frame, track_ids] = 1

                # Get adjacency matrix, cannot filter on track ids.
                cent = centroids[batch, frame]
                distance = cdist(cent, cent, metric='euclidean')
                distance = distance < self.distance_threshold

                # Disconnect the padded nodes
                morphs = morphologies[batch, frame]
                is_pad = np.matmul(morphs, morphs.T) == 0

                adj = distance * (1 - is_pad)
                adj_matrix[batch, frame] = adj.astype(np.float32)

            # Get track length and temporal adjacency matrix
            for label in self.lineages[batch]:
                # Get track length
                start_frame = self.lineages[batch][label]['frames'][0]
                end_frame = self.lineages[batch][label]['frames'][-1]

                track_id = label - 1
                track_length[batch, track_id, 0] = start_frame
                track_length[batch, track_id, 1] = end_frame

                # Get temporal adjacency matrix
                frames = self.lineages[batch][label]['frames']

                # Assign same
                for f0, f1 in zip(frames[0:-1], frames[1:]):
                    if f1 - f0 == 1:
                        temporal_adj_matrix[batch, f0, track_id, track_id, 0] = 1

                # Assign daughter
                # WARNING: This wont work if there's a time gap between mother
                # cell disappearing and daughter cells appearing
                last_frame = frames[-1]
                daughters = self.lineages[batch][label]['daughters']
                for daughter in daughters:
                    daughter_id = daughter - 1
                    temporal_adj_matrix[batch, last_frame, track_id, daughter_id, 2] = 1

            # Assign different
            same_prob = temporal_adj_matrix[batch, ..., 0]
            daughter_prob = temporal_adj_matrix[batch, ..., 2]
            temporal_adj_matrix[batch, ..., 1] = 1 - same_prob - daughter_prob

            # Identify cell padding
            for i in range(temporal_adj_matrix.shape[2]):
                # index + 1 is the cell label
                if i + 1 not in self.lineages[batch]:
                    temporal_adj_matrix[batch, :, i] = 0
                    temporal_adj_matrix[batch, :, :, i] = 0

            # Identify temporal padding
            for b in range(temporal_adj_matrix.shape[0]):
                sames = temporal_adj_matrix[b, ..., 0]
                sames = np.sum(sames, axis=(1, 2))
                temporal_adj_matrix[b, sames == 0] = 0

        features = {
            'adj_matrix': adj_matrix,
            'appearances': appearances,
            'morphologies': morphologies,
            'centroids': centroids,
            'temporal_adj_matrix': temporal_adj_matrix,
            'mask': mask,
            'track_length': track_length,
        }

        return features


def concat_tracks(tracks):
    """Join an iterable of Track objects into a single dictionary of features.
    Args:
        tracks (iterable): Iterable of tracks.
    Returns:
        dict: A dictionary of tracked features.
    Raises:
        TypeError: ``tracks`` is not iterable.
    """
    try:
        list(tracks)  # check if iterable
    except TypeError:
        raise TypeError('concatenate_tracks requires an iterable input.')

    def get_array_of_max_shape(lst):
        # find max dimensions of all arrs in lst.
        shape = None
        size = 0
        for arr in lst:
            if shape is None:
                shape = [0] * len(arr.shape[1:])
            for i, dim in enumerate(arr.shape[1:]):
                if dim > shape[i]:
                    shape[i] = dim
            size += arr.shape[0]
        # add batch dimension
        shape = [size] + shape
        return np.zeros(shape, dtype='float32')

    # insert small array into larger array
    # https://stackoverflow.com/a/50692782
    def paste_slices(tup):
        pos, w, max_w = tup
        wall_min = max(pos, 0)
        wall_max = min(pos + w, max_w)
        block_min = -min(pos, 0)
        block_max = max_w - max(pos + w, max_w)
        block_max = block_max if block_max != 0 else None
        return slice(wall_min, wall_max), slice(block_min, block_max)

    def paste(wall, block, loc):
        loc_zip = zip(loc, block.shape, wall.shape)
        wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
        wall[wall_slices] = block[block_slices]

    def concat_adj_matrices(tracks, matrix_name):
        adj_list = [getattr(t, matrix_name) for t in tracks]
        return tf.sparse.concat(0, adj_list,
                                expand_nonconcat_dims=True)

    # TODO: these keys must match the Track attributes.
    data_dict = {
        'appearances': get_array_of_max_shape((t.appearances for t in tracks)),
        'centroids': get_array_of_max_shape((t.centroids for t in tracks)),
        'morphologies': get_array_of_max_shape((t.morphologies for t in tracks)),
    }

    for track in tracks:
        for k in data_dict:
            feature = getattr(track, k)
            paste(data_dict[k], feature, (0,) * len(feature.shape))

    # handle adj matrices differently as they can be directly concatenated.
    adj_matrix_names = ('adj_matrices', 'norm_adj_matrices', 'temporal_adj_matrices')
    for matrix_name in adj_matrix_names:
        data_dict[matrix_name] = concat_adj_matrices(tracks, matrix_name)

    return data_dict


def temporal_slice(X, y, track_length=8):
    """Randomly slice movies and labels with a length of ``track_length``.

    Args:
        X (dict): Dictionary of feature data.
        y (dict): Dictionary of labels.
        track_length (int): Length of temporal slices.

    Returns:
        tuple(dict, dict): Tuple of sliced ``X`` and ``y`` data.
    """
    temporal_adj_matrices = y['temporal_adj_matrices']

    temporal_adj_matrices = tf.cast(temporal_adj_matrices, tf.int32)

    # Identify max time, accounting for padding
    # Padding frames have zero value accross all channels - look for these
    tam_reduce_sum = tf.sparse.reduce_sum(temporal_adj_matrices, axis=[1, 2, 3])
    non_pad_indices = tf.where(tam_reduce_sum != 0)
    max_time = tf.reduce_max(non_pad_indices)

    max_time = tf.cond(max_time > track_length,
                       lambda: tf.cast(max_time - track_length, tf.int32),
                       lambda: tf.cast(1, tf.int32))

    t_start = tf.random.uniform(shape=[], minval=0,
                                maxval=max_time,
                                dtype=tf.int32)

    t_end = t_start + track_length

    def slice_sparse(sp, t_start, t_length):
        shape = sp.shape.as_list()
        n_dim = len(shape)
        start = [t_start] + [0] * (n_dim - 1)
        size = [t_length] + shape[1:]
        sp_slice = tf.sparse.slice(sp, start=start, size=size)
        return tf.sparse.to_dense(sp_slice)

    for key, data in X.items():
        if isinstance(data, tf.sparse.SparseTensor):
            X[key] = slice_sparse(data, t_start, track_length)
        else:
            X[key] = data[t_start:t_end]

    for key, data in y.items():
        if isinstance(data, tf.sparse.SparseTensor):
            y[key] = slice_sparse(data, t_start, track_length - 1)
        else:
            y[key] = data[t_start:t_end]

    return (X, y)


def random_rotate(X, y, rotation_range=0):
    """Randomly rotate centroids and appearances.

    Args:
        X (dict): Dictionary of feature data.
        rotation_range (int): Maximum rotation range in degrees.

    Returns:
        dict: Rotated ``X`` data.
    """
    appearances = X['appearances']
    centroids = X['centroids']

    # Calculate the random rotation in radians
    rg = rotation_range * math.pi / 180
    theta = tf.random.uniform(shape=[1], minval=-rg, maxval=rg)

    # Transform appearances
    old_shape = tf.shape(appearances)
    new_shape = [-1, old_shape[2], old_shape[3], old_shape[4]]
    img = tf.reshape(appearances, new_shape)
    img = tfa.image.rotate(img, theta)
    img = tf.reshape(img, old_shape)
    X['appearances'] = img

    # Rotate coordinates
    cos_theta = tf.math.cos(theta)
    sin_theta = tf.math.sin(theta)
    rotation_matrix = tf.concat([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ], axis=1)
    transformed_centroids = tf.matmul(centroids, rotation_matrix)
    X['centroids'] = transformed_centroids

    return X, y


def random_translate(X, y, range=512):
    """Randomly translate the centroids."""
    centroids = X['centroids']
    r0 = tf.random.uniform([1, 1, 2], -range, range)
    transformed_centroids = centroids + r0
    X['centroids'] = transformed_centroids
    return X, y


def prepare_dataset(track_info,
                    batch_size=32, buffer_size=256,
                    seed=None, track_length=8, rotation_range=0,
                    translation_range=0, val_size=0.2, test_size=0):
    """Build and prepare the tracking dataset.

    Args:
        track_info (dict): A dictionary of all input and output features
        batch_size (int): number of examples per batch
        buffer_size (int): number of samples to buffer
        seed (int): Random seed
        track_length (int): Number of frames per example
        rotation_range (int): Maximum degrees to rotate inputs
        translation_range (int): Maximum range of translation,
            should be equivalent to original input image size.
        val_size (float): Fraction of data to split into validation set.
        test_size (float): Fraction of data to split into test set.

    Returns:
        tf.data.Dataset: A ``tf.data.Dataset`` object ready for training.
    """
    input_dict = {
        'appearances': track_info['appearances'],
        'centroids': track_info['centroids'],
        'morphologies': track_info['morphologies'],
        'adj_matrices': track_info['norm_adj_matrices'],
    }

    output_dict = {'temporal_adj_matrices': track_info['temporal_adj_matrices']}

    dataset = tf.data.Dataset.from_tensor_slices((input_dict, output_dict))

    dataset = dataset.shuffle(buffer_size, seed=seed).repeat()

    # randomly sample along the temporal axis
    sample = lambda X, y: temporal_slice(X, y, track_length=track_length)
    dataset = dataset.map(sample, num_parallel_calls=tf.data.AUTOTUNE)

    # split into train/val before doing any augmentation
    train_data, val_data, test_data = split_dataset(dataset, val_size, test_size)

    # randomly rotate
    rotate = lambda X, y: random_rotate(X, y, rotation_range=rotation_range)
    train_data = train_data.map(rotate, num_parallel_calls=tf.data.AUTOTUNE)

    # randomly translate centroids
    translate = lambda X, y: random_translate(X, y, range=translation_range)
    train_data = train_data.map(translate, num_parallel_calls=tf.data.AUTOTUNE)

    # batch the data
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    # prefetch the data
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.AUTOTUNE)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)

    return train_data, val_data, test_data
