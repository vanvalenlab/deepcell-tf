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
"""Dataset Builders"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class TrackingDatasetBuilder(object):
    """Create a tf.data.Dataset object for training tracking models.

    Args:
        track_list (list): A list of `track` objects containing lineage info
        track_length (int): Number of desired frames per track
        batch_size (int): Number of desired batches per dataset
    """
    def __init__(self,
                 track_list,
                 track_length=8,
                 batch_size=1):
        self.batch_size = batch_size
        self.track_list = track_list
        self.track_length = track_length

        # Create max_nodes and max_frames
        self.max_nodes, self.max_frames = self._get_max_sizes()

        # Load tracks
        track_info = self._load_tracks()
        self.appearances = track_info['appearances']
        self.centroids = track_info['centroids']
        self.morphologies = track_info['morphologies']
        self.adj_matrices = track_info['adj_matrices']
        self.norm_adj_matrices = track_info['norm_adj_matrices']
        self.temporal_adj_matrices = track_info['temporal_adj_matrices']

        # Create datasets
        self.dataset = self._create_dataset()

    def _get_max_sizes(self):
        n_nodes = [track.appearances.shape[1] for track in self.track_list]
        max_nodes = max(n_nodes)

        n_frames = [track.appearances.shape[2] for track in self.track_list]
        max_frames = max(n_frames)

        return max_nodes, max_frames

    def _pad_array(self,
                   arr,
                   node_axes=[1],
                   time_axes=[2],
                   temporal_adj=False,
                   pad_value=0):

        n_axes = len(arr.shape)
        pads = []

        max_nodes = self.max_nodes
        max_frames = self.max_frames

        if temporal_adj:
            max_frames -= 1

        for ax in range(n_axes):
            if ax in node_axes:
                pads.append((0, max_nodes - arr.shape[ax]))
            elif ax in time_axes:
                pads.append((0, max_frames - arr.shape[ax]))
            else:
                pads.append((0, 0))
        pads = tuple(pads)
        values = (pad_value, pad_value)
        arr = np.pad(arr,
                     pads,
                     mode='constant',
                     constant_values=values)
        return arr

    def _load_tracks(self):

        appearances = []
        centroids = []
        morphologies = []
        adj_matrices = []
        norm_adj_matrices = []
        temporal_adj_matrices = []

        for track in self.track_list:
            app = self._pad_array(track.appearances)
            cent = self._pad_array(track.centroids)
            morph = self._pad_array(track.morphologies)
            adj = self._pad_array(track.adj_matrices,
                                  node_axes=[1, 2],
                                  time_axes=[3])
            norm_adj = self._pad_array(track.norm_adj_matrices,
                                       node_axes=[1, 2],
                                       time_axes=[3])
            temporal_adj = self._pad_array(track.temporal_adj_matrices,
                                           temporal_adj=True,
                                           node_axes=[1, 2],
                                           time_axes=[3])

            appearances.append(app)
            centroids.append(cent)
            morphologies.append(morph)
            adj_matrices.append(adj)
            norm_adj_matrices.append(norm_adj)
            temporal_adj_matrices.append(temporal_adj)

        track_info = {}
        track_info['appearances'] = np.concatenate(appearances, axis=0)
        track_info['centroids'] = np.concatenate(centroids, axis=0)
        track_info['morphologies'] = np.concatenate(morphologies, axis=0)
        track_info['adj_matrices'] = np.concatenate(adj_matrices, axis=0)
        track_info['norm_adj_matrices'] = np.concatenate(norm_adj_matrices, axis=0)
        track_info['temporal_adj_matrices'] = np.concatenate(temporal_adj_matrices, axis=0)

        return track_info

    def _sample_time(self, *args):
        X_dict = args[0]
        y_dict = args[1]

        app = X_dict['appearances']
        max_time = tf.shape(app)[1] - self.track_length
        t_start = tf.random.uniform([1],
                                    minval=0,
                                    maxval=max_time,
                                    dtype=tf.int32)[0]

        for key in X_dict:
            data = X_dict[key]
            if 'adj' not in key:
                sampled_data = data[:, t_start:t_start + self.track_length, ...]
                X_dict[key] = sampled_data
            else:
                sampled_data = data[:, :, t_start:t_start + self.track_length]
                X_dict[key] = sampled_data

        for key in y_dict.keys():
            data = y_dict[key]
            sampled_data = data[:, :, t_start:t_start + self.track_length - 1, ...]
            y_dict[key] = sampled_data

        return (X_dict, y_dict)

    def _augment(self, *args):
        X_dict = args[0]
        y_dict = args[1]

        app = X_dict['appearances']
        centroids = X_dict['centroids']

        # Randomly rotate appearances
        theta = tf.random.uniform([1], 0, 2 * 3.1415926)

        # Transform appearances
        old_shape = tf.shape(app)
        new_shape = [-1, tf.shape(app)[2], tf.shape(app)[3], tf.shape(app)[4]]
        img = tf.reshape(app, new_shape)
        img = tfa.image.rotate(img, theta)
        img = tf.reshape(img, old_shape)

        X_dict['appearances'] = img

        # Transform coordinates
        cos_theta = tf.math.cos(theta)
        sin_theta = tf.math.sin(theta)

        rot_row_0 = tf.stack([cos_theta, -sin_theta], axis=1)
        rot_row_1 = tf.stack([sin_theta, cos_theta], axis=1)
        rotation_matrix = tf.concat([rot_row_0, rot_row_1], axis=0)
        transformed_centroids = tf.matmul(centroids, tf.transpose(rotation_matrix))
        r0 = tf.random.uniform([1, 1, 2], -512, 512)
        transformed_centroids = transformed_centroids + r0
        X_dict['centroids'] = transformed_centroids

        return (X_dict, y_dict)

    def _create_dataset(self):
        input_dict = {'appearances': self.appearances,
                      'centroids': self.centroids,
                      'morphologies': self.morphologies,
                      'adj_matrices': self.norm_adj_matrices}
        output_dict = {'temporal_adj_matrices': self.temporal_adj_matrices}
        dataset = tf.data.Dataset.from_tensor_slices((input_dict, output_dict))
        dataset = dataset.shuffle(256).repeat().map(self._sample_time).map(self._augment).batch(self.batch_size)

        return dataset

# TODO: Include a split dataset utils module when ready
# TODO: the train/test/val split should be addressed
