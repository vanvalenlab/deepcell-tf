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
"""A cell tracking class capable of extending labels across sequential frames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import pathlib
import tarfile
import tempfile
import timeit

import pandas as pd
import networkx as nx

import cv2
import numpy as np
from tensorflow.python.keras import backend as K
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.transform import resize


class cell_tracker(object):  # pylint: disable=useless-object-inheritance

    def __init__(self,
                 movie,
                 annotation,
                 model,
                 features=None,
                 crop_dim=32,
                 death=0.9,
                 birth=0.9,
                 division=0.2,
                 max_distance=200,
                 track_length=1,
                 neighborhood_scale_size=10,
                 neighborhood_true_size=100,
                 data_format=None):

        if not len(movie.shape) == 4 or not len(annotation.shape) == 4:
            raise ValueError('Input data and labels but be rank 4 '
                             '(frames, x, y, channels).  Got {} and {}.'.format(
                                 len(movie.shape), len(annotation.shape)))

        if not movie.shape[:-1] == annotation.shape[:-1]:
            raise ValueError('Input data and labels should have the same shape'
                             ' except for the channel dimension.  Got {} and '
                             '{}'.format(movie.shape, annotation.shape))

        if data_format is None:
            data_format = K.image_data_format()

        if features is None:  # TODO: why default to None?
            raise ValueError('cell_tracking: No features specified.')

        self.x = copy.copy(movie)
        self.y = copy.copy(annotation)
        # TODO: Use a model that is served by tf-serving, not one on a local machine
        self.model = model
        self.crop_dim = crop_dim
        self.death = death
        self.birth = birth
        self.division = division
        self.max_distance = max_distance
        self.neighborhood_scale_size = neighborhood_scale_size
        self.neighborhood_true_size = neighborhood_true_size
        self.data_format = data_format
        self.track_length = track_length
        self.channel_axis = 0 if data_format == 'channels_first' else -1

        self.features = sorted(features)
        self.feature_shape = {
            'appearance': (crop_dim, crop_dim, self.x.shape[self.channel_axis]),
            'neighborhood': (2 * neighborhood_scale_size + 1,
                             2 * neighborhood_scale_size + 1, 1),
            'regionprop': (3,),
            'distance': (2,),
        }

        # Clean up annotations
        self._clean_up_annotations()

        # Initialize tracks
        self._initialize_tracks()

    def _clean_up_annotations(self):
        """Relabels every frame in the label matrix.
        Cells will be relabeled 1 to N
        """
        y = self.y
        number_of_frames = self.y.shape[0]

        # The annotations need to be unique across all frames
        # TODO: Resolve the starting unique ID issue
        uid = 1000
        for frame in range(number_of_frames):
            unique_cells = np.unique(y[frame])
            y_frame_new = np.zeros(y[frame].shape)
            for _, old_label in enumerate(list(unique_cells)):
                if old_label == 0:
                    y_frame_new[y[frame] == old_label] = 0
                else:
                    y_frame_new[y[frame] == old_label] = uid
                    uid += 1
            y[frame] = y_frame_new
        self.y = y.astype('int32')

    def _create_new_track(self, frame, old_label):
        """
        This function creates new tracks
        """

        new_track = len(self.tracks.keys())
        new_label = new_track + 1

        self.tracks[new_track] = {}
        self.tracks[new_track]['label'] = new_label

        self.tracks[new_track]['frames'] = [frame]
        self.tracks[new_track]['daughters'] = []
        self.tracks[new_track]['capped'] = False
        self.tracks[new_track]['frame_div'] = None
        self.tracks[new_track]['parent'] = None

        self.tracks[new_track].update(self._get_features(self.x, self.y, [frame], [old_label]))

        if frame > 0 and np.any(self.y[frame] == new_label):
            raise Exception('new_label already in annotated frame and frame > 0')

        self.y[frame][self.y[frame] == old_label] = new_label

    def _initialize_tracks(self):
        """Intialize the tracks. Tracks are stored in a dictionary.
        """
        self.tracks = {}
        unique_cells = np.unique(self.y[0])

        # Remove background that has value 0
        unique_cells = np.delete(unique_cells, np.where(unique_cells == 0))

        for track_counter, label in enumerate(unique_cells):
            self._create_new_track(0, label)

        # Start a tracked label array
        self.y_tracked = self.y[[0]].astype('int32')

    def _compute_feature(self, feature_name, track_feature, frame_feature):
        """
        Given a track and frame feature, compute the resulting track and frame features.
        This also returns True or False as the third element of the tuple indicating if these
        features should be used at all. False indicates that this pair of track & cell features
        should result in a maximum cost assignment.
        This is usually for some preprocessing in case it is desired. For example, the
        distance feature normalizes distances.
        """
        if feature_name == 'appearance':
            return track_feature, frame_feature, True

        if feature_name == 'distance':
            centroids = np.concatenate([track_feature, np.array([frame_feature])], axis=0)
            distances = np.diff(centroids, axis=0)
            zero_pad = np.zeros((1, 2), dtype=K.floatx())
            distances = np.concatenate([zero_pad, distances], axis=0)

            ok = True
            # Make sure the distances are all less than max distance
            for j in range(distances.shape[0]):
                dist = distances[j, :]
                # print('distance: ', np.linalg.norm(dist))  # TODEL
                # TODO(enricozb): Finish the distance-based optimizations
                if np.linalg.norm(dist) > self.max_distance:
                    ok = False
            return distances[0:-1, :], distances[-1, :], ok

        if feature_name == 'neighborhood':
            return track_feature, frame_feature, True

        if feature_name == 'regionprop':
            return track_feature, frame_feature, True

        raise ValueError('_fetch_track_feature: '
                         'Unknown feature `{}`'.format(feature_name))

    def _get_cost_matrix(self, frame):
        """Uses the model to create the cost matrix for
        assigning the cells in frame to existing tracks.
        """
        t = timeit.default_timer()
        # Initialize matrices
        number_of_tracks = np.int(len(self.tracks.keys()))

        cells_in_frame = np.unique(self.y[frame])
        cells_in_frame = list(np.delete(cells_in_frame, np.where(cells_in_frame == 0)))
        number_of_cells = len(cells_in_frame)

        total_cells = number_of_tracks + number_of_cells
        cost_matrix = np.zeros((total_cells, total_cells), dtype=K.floatx())
        assignment_matrix = np.zeros((number_of_tracks, number_of_cells), dtype=K.floatx())
        birth_matrix = np.zeros((number_of_cells, number_of_cells), dtype=K.floatx())
        death_matrix = np.zeros((number_of_tracks, number_of_tracks), dtype=K.floatx())

        # Bottom right matrix
        mordor_matrix = np.zeros((number_of_cells, number_of_tracks), dtype=K.floatx())

        # Grab the features for the entire track
        track_features = {f: self._fetch_track_feature(f) for f in self.features}

        # Grab the features for this frame
        # Fill frame_features with zero matrices
        frame_features = {}
        for feature_name in self.features:
            feature_shape = self.feature_shape[feature_name]
            # TODO(enricozb): why are there extra (1,)'s in the image shapes
            additional = (1,) if feature_name in {'appearance', 'neighborhood'} else ()
            shape = tuple([number_of_cells] + list(additional) + list(feature_shape))
            frame_features[feature_name] = np.zeros(shape, dtype=K.floatx())
        # Fill frame_features with the proper values
        for cell_idx, cell_id in enumerate(cells_in_frame):
            cell_features = self._get_features(self.x, self.y, [frame], [cell_id])
            for feature_name in self.features:
                frame_features[feature_name][cell_idx] = cell_features[feature_name]

        # Call model.predict only on inputs that are near each other
        inputs = {feature_name: ([], []) for feature_name in self.features}
        input_pairs = []

        # Compute assignment matrix - Initialize and get model inputs
        # Fill the input matrices
        for track in range(number_of_tracks):

            # we need to get the future frame for the track we are comparing to
            try:
                track_label = self.tracks[track]['label']
                track_frame_features = self._get_features(
                    self.x, self.y_tracked, [frame - 1], [track_label])
            except:
                # `track_label` might not exist in `frame - 1`
                # if this happens, default to the cell's neighborhood
                track_frame_features = dict()

            for cell in range(number_of_cells):
                feature_vals = {}

                # If distance is a feature it is used to exclude
                # impossible pairings from the get_feature call
                if 'distance' in self.features:
                    _, _, is_cell_in_range = self._compute_feature(
                        'distance',
                        track_features['distance'][track],
                        frame_features['distance'][cell])
                else:
                    # not worried about distance, just calculate features
                    is_cell_in_range = True

                if not is_cell_in_range:
                    # Cell is outside of range, set cost to max and move on
                    assignment_matrix[track, cell] = 1
                    continue

                # The cell is within range so we should add
                # all the information for all features
                for feature_name in self.features:

                    track_feature, frame_feature, _ = self._compute_feature(
                        feature_name,
                        track_features[feature_name][track],
                        frame_features[feature_name][cell])

                    # this condition changes `frame_feature`
                    if feature_name == 'neighborhood':
                        # This segment of the loop should not be run
                        # if the disance check fails
                        frame_feature = track_frame_features.get('~future area', frame_feature)

                    feature_vals[feature_name] = (track_feature, frame_feature)

                input_pairs.append((track, cell))
                for feature_name, (track_feature, frame_feature) in feature_vals.items():
                    inputs[feature_name][0].append(track_feature)
                    inputs[feature_name][1].append(frame_feature)

        print('Got features in {}s'.format(timeit.default_timer() - t))

        if input_pairs == []:
            # if the frame is empty
            assignment_matrix[:, :] = 1
            predictions = []
        else:
            model_input = []
            for feature_name in self.features:
                in_1, in_2 = inputs[feature_name]
                feature_shape = self.feature_shape[feature_name]
                in_1 = np.reshape(np.stack(in_1),
                                  tuple([len(input_pairs), self.track_length] +
                                        list(feature_shape)))
                in_2 = np.reshape(np.stack(in_2), tuple([len(input_pairs), 1] +
                                                        list(feature_shape)))
                model_input.extend([in_1, in_2])

            predictions = self.model.predict(model_input)

            for i, (track, cell) in enumerate(input_pairs):
                assignment_matrix[track, cell] = 1 - predictions[i, 1]

        # Make sure capped tracks are not allowed to have assignments
        for track in range(number_of_tracks):
            if self.tracks[track]['capped']:
                assignment_matrix[track, 0:number_of_cells] = 1

        # Compute birth matrix
        birth_diagonal = np.array([self.birth] * number_of_cells)
        birth_matrix = np.diag(birth_diagonal) + np.ones(birth_matrix.shape)
        birth_matrix = birth_matrix - np.eye(number_of_cells)

        # Compute death matrix
        death_matrix = self.death * np.eye(number_of_tracks) + np.ones(death_matrix.shape)
        death_matrix = death_matrix - np.eye(number_of_tracks)

        # Compute mordor matrix
        mordor_matrix = assignment_matrix.T

        # Assemble full cost matrix
        cost_matrix[0:number_of_tracks, 0:number_of_cells] = assignment_matrix
        cost_matrix[number_of_tracks:, 0:number_of_cells] = birth_matrix
        cost_matrix[0:number_of_tracks, number_of_cells:] = death_matrix
        cost_matrix[number_of_tracks:, number_of_cells:] = mordor_matrix

        predictions_map = {pair: prediction
                           for pair, prediction in zip(input_pairs, predictions)}

        return cost_matrix, predictions_map

    def _run_lap(self, cost_matrix):
        """Runs the linear assignment function on a cost matrix.
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = np.stack([row_ind, col_ind], axis=1)

        return assignments

    def _update_tracks(self, assignments, frame, predictions):
        """Update the tracks if given the assignment matrix
        and the frame that was tracked.
        """
        number_of_tracks = len(self.tracks.keys())
        cells_in_frame = np.unique(self.y[frame])
        cells_in_frame = np.delete(cells_in_frame, np.where(cells_in_frame == 0))
        # Number of lables present in the current frame (needed to build cost matrix)
        number_of_cells = len(list(cells_in_frame))
        y_tracked_update = np.zeros((1, self.y.shape[1], self.y.shape[2], 1), dtype='int32')

        for a in range(assignments.shape[0]):
            track, cell = assignments[a]
            track_id = track + 1  # Labels and indices differ by 1

            # This is a mapping of the column index provided by the lap
            # assignment to the cell label in the frame
            if cell < number_of_cells:
                cell_id = cells_in_frame[cell]  # This is the new mapping

            # Take care of everything if cells are tracked
            if track < number_of_tracks and cell < number_of_cells:
                self.tracks[track]['frames'].append(frame)
                cell_features = self._get_features(self.x, self.y, [frame], [cell_id])
                for feature_name, cell_feature in cell_features.items():
                    self.tracks[track][feature_name] = np.concatenate([
                        self.tracks[track][feature_name], cell_feature], axis=0)

                y_tracked_update[self.y[[frame]] == cell_id] = track_id
                self.y[frame][self.y[frame] == cell_id] = track_id

            # Create a new track if there was a birth
            elif track > number_of_tracks - 1 and cell < number_of_cells:
                new_track_id = len(self.tracks.keys())
                self._create_new_track(frame, cell_id)
                new_label = new_track_id + 1

                # See if the new track has a parent
                parent = self._get_parent(frame, cell, predictions)
                if parent is not None:
                    print('Division detected')
                    self.tracks[new_track_id]['parent'] = parent
                    self.tracks[parent]['daughters'].append(new_track_id)
                else:
                    self.tracks[new_track_id]['parent'] = None

                y_tracked_update[self.y[[frame]] == new_label] = new_track_id + 1
                self.y[frame][self.y[frame] == new_label] = new_track_id + 1

            # Dont touch anything if there was a cell that "died"
            elif track < number_of_tracks and cell > number_of_cells - 1:
                continue

        # Cap the tracks of cells that divided
        number_of_tracks = len(self.tracks.keys())
        for track in range(number_of_tracks):
            if self.tracks[track]['daughters'] and not self.tracks[track]['capped']:
                self.tracks[track]['frame_div'] = int(frame)
                self.tracks[track]['capped'] = True

        # Check and make sure cells that divided did not get assigned to the same cell
        for track in range(number_of_tracks):
            if self.tracks[track]['daughters']:
                if frame in self.tracks[track]['frames']:
                    # Create new track
                    old_label = self.tracks[track]['label']
                    new_track_id = len(self.tracks.keys())
                    new_label = new_track_id + 1
                    self._create_new_track(frame, old_label)

                    for feature_name in self.features:
                        fname = self.tracks[track][feature_name][[-1]]
                        self.tracks[new_track_id][feature_name] = fname

                    self.tracks[new_track_id]['parent'] = track

                    # Remove frame from old track
                    self.tracks[track]['frames'].remove(frame)
                    for feature_name in self.features:
                        fname = self.tracks[track][feature_name][0:-1]
                        self.tracks[track][feature_name] = fname
                    self.tracks[track]['daughters'].append(new_track_id)

                    # Change y_tracked_update
                    y_tracked_update[self.y[[frame]] == new_label] = new_track_id + 1
                    self.y[frame][self.y[frame] == new_label] = new_track_id + 1

        # Update the tracked label array
        self.y_tracked = np.concatenate([self.y_tracked, y_tracked_update], axis=0)

    def _get_parent(self, frame, cell, predictions):
        """Searches the tracks for the parent of a given cell.

        Returns:
            The parent cell's id or None if no parent exists.
        """
        # are track_ids 0-something or 1-something??
        # 0-something because of the `for track_id, p in enumerate(...)` below
        probs = {}
        for (track, cell_id), p in predictions.items():
            # Make sure capped tracks can't be assigned parents
            if cell_id == cell and not self.tracks[track]['capped']:
                probs[track] = p[2]

        # Find out if the cell is a daughter of a track
        print('New track')
        max_prob = self.division
        parent_id = None
        for track_id, p in probs.items():
            # we don't want to think a sibling of `cell`, that just appeared
            # is a parent
            if self.tracks[track_id]['frames'] == [frame]:
                continue
            if p > max_prob:
                parent_id, max_prob = track_id, p
        return parent_id

    def _fetch_track_feature(self, feature, before_frame=None):
        if before_frame is None:
            before_frame = float('inf')

        if feature == 'appearance':
            return self._fetch_track_appearances(before_frame)
        if feature == 'distance':
            return self._fetch_track_centroids(before_frame)
        if feature == 'regionprop':
            return self._fetch_track_regionprops(before_frame)
        if feature == 'neighborhood':
            return self._fetch_track_neighborhoods(before_frame)

        raise ValueError('_fetch_track_feature: '
                         'Unknown feature `{}`'.format(feature))

    def _fetch_track_appearances(self, before_frame):
        """
        This function fetches the appearances for all of the existing tracks.
        If tracks are shorter than the track length, they are filled in with
        the first frame.
        """
        shape = (len(self.tracks.keys()),
                 self.track_length,
                 self.crop_dim,
                 self.crop_dim,
                 self.x.shape[self.channel_axis])
        track_appearances = np.zeros(shape, dtype=K.floatx())

        for track_id, track in self.tracks.items():
            app = track['appearance']
            allowed_frames = [f for f in track['frames'] if f < before_frame]
            frame_dict = {frame: idx for idx, frame in enumerate(allowed_frames)}

            if not allowed_frames:
                continue

            if len(allowed_frames) >= self.track_length:
                frames = allowed_frames[-self.track_length:]
            else:
                num_missing = self.track_length - len(allowed_frames)
                last_frame = allowed_frames[-1]
                frames = allowed_frames + [last_frame] * num_missing

            track_appearances[track_id] = app[[frame_dict[f] for f in frames]]

        return track_appearances

    def _fetch_track_regionprops(self, before_frame):
        """Fetches the regionprops for all of the existing tracks.
        If tracks are shorter than the track length they are filled in with
        the centroids from the first frame.
        """
        shape = (len(self.tracks.keys()), self.track_length, 3)
        track_regionprops = np.zeros(shape, dtype=K.floatx())

        for track_id, track in self.tracks.items():
            regionprop = track['regionprop']
            allowed_frames = list(filter(lambda f: f < before_frame, track['frames']))
            frame_dict = {frame: idx for idx, frame in enumerate(allowed_frames)}

            if not allowed_frames:
                continue

            if len(allowed_frames) >= self.track_length:
                frames = allowed_frames[-self.track_length:]
            else:
                num_missing = self.track_length - len(allowed_frames)
                last_frame = allowed_frames[-1]
                frames = allowed_frames + [last_frame] * num_missing

            track_regionprops[track_id] = regionprop[[frame_dict[f] for f in frames]]

        return track_regionprops

    def _fetch_track_centroids(self, before_frame):
        """Fetches the centroids for all of the existing tracks.
        If tracks are shorter than the track length they are filled in with
        the centroids from the first frame.
        """
        shape = (len(self.tracks.keys()), self.track_length, 2)
        track_centroids = np.zeros(shape, dtype=K.floatx())

        for track_id, track in self.tracks.items():
            centroids = track['distance']
            allowed_frames = list(filter(lambda f: f < before_frame, track['frames']))
            frame_dict = {frame: idx for idx, frame in enumerate(allowed_frames)}

            if len(allowed_frames) == 0:
                continue

            if len(allowed_frames) >= self.track_length:
                frames = allowed_frames[-self.track_length:]
            else:
                num_missing = self.track_length - len(allowed_frames)
                last_frame = allowed_frames[-1]
                frames = allowed_frames + [last_frame] * num_missing

            track_centroids[track_id] = centroids[[frame_dict[f] for f in frames]]

        return track_centroids

    def _fetch_track_neighborhoods(self, before_frame):
        """Gets the neighborhoods for all of the existing tracks.
        If tracks are shorter than the track length they are filled in with the
        neighborhoods from the first frame.
        """
        shape = (len(self.tracks.keys()),
                 self.track_length,
                 2 * self.neighborhood_scale_size + 1,
                 2 * self.neighborhood_scale_size + 1,
                 1)
        track_neighborhoods = np.zeros(shape, dtype=K.floatx())

        for track_id, track in self.tracks.items():
            neighborhoods = track['neighborhood']
            allowed_frames = list(filter(lambda f: f < before_frame, track['frames']))
            frame_dict = {frame: idx for idx, frame in enumerate(allowed_frames)}

            if len(allowed_frames) == 0:
                continue

            if len(allowed_frames) >= self.track_length:
                frames = allowed_frames[-self.track_length:]
            else:
                num_missing = self.track_length - len(allowed_frames)
                last_frame = allowed_frames[-1]
                frames = allowed_frames + [last_frame] * num_missing

            track_neighborhoods[track_id] = neighborhoods[[frame_dict[f] for f in frames]]

        return track_neighborhoods

    def _sub_area(self, X_frame, y_frame, cell_label, num_channels):
        t = timeit.default_timer()
        true_size = self.neighborhood_true_size
        pads = ((true_size, true_size),
                (true_size, true_size),
                (0, 0))

        X_padded = np.pad(X_frame, pads, mode='constant', constant_values=0)
        y_padded = np.pad(y_frame, pads, mode='constant', constant_values=0)

        props = regionprops(np.squeeze(np.int32(y_padded == cell_label)))

        center_x, center_y = props[0].centroid
        center_x, center_y = np.int(center_x), np.int(center_y)

        X_reduced = X_padded[center_x - true_size:center_x + true_size,
                             center_y - true_size:center_y + true_size]

        # resize to neighborhood_scale_size with skimage
        # resize_shape = (2 * self.neighborhood_scale_size + 1,
        #                 2 * self.neighborhood_scale_size + 1,
        #                 num_channels)
        # X_reduced = resize(X_reduced, resize_shape, mode='constant', preserve_range=True)

        # resize to neighborhood_scale_size with cv2
        resize_shape = (2 * self.neighborhood_scale_size + 1,
                        2 * self.neighborhood_scale_size + 1)
        X_reduced = cv2.resize(np.squeeze(X_reduced), resize_shape)

        # X_reduced /= np.amax(X_reduced)
        X_reduced = np.expand_dims(X_reduced, axis=self.channel_axis)
        print('_sub_area finished in {}s'.format(timeit.default_timer() - t))
        return X_reduced

    def _get_features(self, X, y, frames, labels):
        """Gets the features of a list of cells.
        Cells are defined by lists of frames and labels. The i'th element of
        frames and labels is the frame and label of the i'th cell being grabbed.
        Returns a dictionary with keys as the feature names.
        """
        channel_axis = self.channel_axis
        if self.data_format == 'channels_first':
            appearance_shape = (X.shape[channel_axis],
                                len(frames),
                                self.crop_dim,
                                self.crop_dim)
        else:
            appearance_shape = (len(frames),
                                self.crop_dim,
                                self.crop_dim,
                                X.shape[channel_axis])

        centroid_shape = (len(frames), 2)
        regionprop_shape = (len(frames), 3)

        neighborhood_shape = (len(frames),
                              2 * self.neighborhood_scale_size + 1,
                              2 * self.neighborhood_scale_size + 1, 1)

        # look-ahead neighborhoods
        future_area_shape = (len(frames),
                             2 * self.neighborhood_scale_size + 1,
                             2 * self.neighborhood_scale_size + 1, 1)

        # Initialize storage for appearances and centroids
        appearances = np.zeros(appearance_shape, dtype=K.floatx())
        centroids = np.zeros(centroid_shape, dtype=K.floatx())
        rprops = np.zeros(regionprop_shape, dtype=K.floatx())
        neighborhoods = np.zeros(neighborhood_shape, dtype=K.floatx())
        future_areas = np.zeros(future_area_shape, dtype=K.floatx())
        for counter, (frame, cell_label) in enumerate(zip(frames, labels)):
            print('Start _get_features for frame {} and label {}'.format(
                frame, cell_label))
            t = timeit.default_timer()
            # Get the bounding box
            X_frame = X[frame] if self.data_format == 'channels_last' else X[:, frame]
            y_frame = y[frame] if self.data_format == 'channels_last' else y[:, frame]
            props = regionprops(np.squeeze(np.int32(y_frame == cell_label)))

            minr, minc, maxr, maxc = props[0].bbox
            centroids[counter] = props[0].centroid
            rprops[counter] = np.array([
                props[0].area,
                props[0].perimeter,
                props[0].eccentricity
            ])
            print('Got regionprops data in {}s'.format(timeit.default_timer() - t))

            # Extract images from bounding boxes
            if self.data_format == 'channels_first':
                appearance = np.copy(X[:, frame, minr:maxr, minc:maxc])
                resize_shape = (X.shape[channel_axis], self.crop_dim, self.crop_dim)
            else:
                appearance = np.copy(X[frame, minr:maxr, minc:maxc, :])
                resize_shape = (self.crop_dim, self.crop_dim, X.shape[channel_axis])

            # Resize images from bounding box
            t = timeit.default_timer()
            # appearance = resize(appearance, resize_shape, mode="constant", preserve_range=True)
            resize_shape = (self.crop_dim, self.crop_dim)
            appearance = cv2.resize(np.squeeze(appearance), resize_shape)
            # appearance /= np.amax(appearance)
            appearance = np.expand_dims(appearance, axis=self.channel_axis)

            if self.data_format == 'channels_first':
                appearances[:, counter] = appearance
            else:
                appearances[counter] = appearance

            print('Reshaped appearance in {}s'.format(timeit.default_timer() - t))

            # Get the neighborhood
            neighborhoods[counter] = self._sub_area(
                X_frame, y_frame, cell_label, X.shape[channel_axis])

            # Try to assign future areas if future frame is available
            # TODO: We shouldn't grab a future frame if the frame is dark (was padded)
            try:
                if self.data_format == 'channels_first':
                    X_future_frame = X[:, frame + 1]
                else:
                    X_future_frame = X[frame + 1]
                future_areas[counter] = self._sub_area(
                    X_future_frame, y_frame, cell_label, X.shape[channel_axis])
            except IndexError:
                future_areas[counter] = neighborhoods[counter]

        # future areas are not a feature instead a part of the neighborhood feature
        return {'appearance': appearances,
                'distance': centroids,
                'neighborhood': neighborhoods,
                'regionprop': rprops,
                '~future area': future_areas}

    def _track_cells(self):
        """Tracks all of the cells in every frame.
        """
        for frame in range(1, self.x.shape[0]):
            print('Tracking frame ' + str(frame))

            t_whole = timeit.default_timer()  # TODEL
            t = timeit.default_timer()  # TODEL

            cost_matrix, predictions = self._get_cost_matrix(frame)

            print('Time to get_cost_matrix: ', timeit.default_timer() - t)  # TODEL
            t = timeit.default_timer()  # TODEL

            assignments = self._run_lap(cost_matrix)

            print('Time to run lap: ', timeit.default_timer() - t)  # TODEL
            t = timeit.default_timer()  # TODEL

            self._update_tracks(assignments, frame, predictions)

            print('Time to update tracks: ', timeit.default_timer() - t)  # TODEL
            print('Time to track one frame: ', timeit.default_timer() - t_whole)  # TODEL

    def _track_review_dict(self):
        def process(key, track_item):
            if track_item is None:
                return track_item
            if key == 'daughters':
                return list(map(lambda x: x + 1, track_item))
            elif key == 'parent':
                return track_item + 1
            else:
                return track_item

        track_keys = ['label', 'frames', 'daughters', 'capped', 'frame_div', 'parent']

        return {'tracks': {track['label']: {key: process(key, track[key]) for key in track_keys}
                           for _, track in self.tracks.items()},
                'X': self.x,
                'y': self.y,
                'y_tracked': self.y}

    def dataframe(self, **kwargs):
        """Returns a dataframe of the tracked cells with lineage.
        Uses only the cell labels not the ids.

        _track_cells must be called first!
        """
        # possible kwargs are extra_columns
        extra_columns = ['cell_type', 'set', 'part', 'montage']
        track_columns = ['label', 'daughters', 'frame_div']

        incorrect_args = set(kwargs) - set(extra_columns)
        if incorrect_args:
            raise ValueError('Invalid argument {}'.format(incorrect_args.pop()))

        # filter extra_columns by the ones we passed in
        extra_columns = [c for c in extra_columns if c in kwargs]

        # extra_columns are the same for every row, cache the values
        extra_column_vals = [kwargs[c] for c in extra_columns if c in kwargs]

        # fill the dataframe
        data = []
        for cell_id, track in self.tracks.items():
            data.append(extra_column_vals + [track[c] for c in track_columns])
        dataframe = pd.DataFrame(data, columns=extra_columns + track_columns)

        # daughters contains track_id not labels
        dataframe['daughters'] = dataframe['daughters'].apply(
            lambda d: [self.tracks[x]['label'] for x in d])

        return dataframe

    def postprocess(self, filename=None, time_excl=9):
        """Use graph postprocessing to eliminate false positive division errors
        using a graph-based detection method. False positive errors are when a
        cell is noted as a daughter of itself before the actual division occurs.
        If a filename is passed, save the state of the cell tracker to a .trk
        ('track') file. time_excl is the minimum number of frames expected to
        exist between legitimate divisions
        """

        # Load data
        track_review_dict = self._track_review_dict()

        # Prep data
        tracked = track_review_dict['y_tracked'].astype('uint16')
        lineage = track_review_dict['tracks']

        # Identify false positives (FPs)
        G = self._track_to_graph(lineage)
        FPs = self._flag_false_pos(G, time_excl)
        FPs_candidates = sorted(FPs.items(), key=lambda v: int(v[0].split('_')[1]))
        FPs_sorted = self._review_candidate_nodes(FPs_candidates)

        # If FPs exist, use the results to correct
        while len(FPs_sorted) != 0:

            lineage, tracked = self._remove_false_pos(lineage, tracked, FPs_sorted[0])
            G = self._track_to_graph(lineage)
            FPs = self._flag_false_pos(G, time_excl)
            FPs_candidates = sorted(FPs.items(), key=lambda v: int(v[0].split('_')[1]))
            FPs_sorted = self._review_candidate_nodes(FPs_candidates)

        # Make sure the assignment is correct
        track_review_dict['y_tracked'] = tracked
        track_review_dict['tracks'] = lineage

        # Save information to a track file file if requested
        if filename is not None:
            # Prep filepath
            filename = pathlib.Path(filename)
            if filename.suffix != '.trk':
                filename = filename.with_suffix('.trk')

            filename = str(filename)

            # Save
            with tarfile.open(filename, 'w') as trks:
                with tempfile.NamedTemporaryFile('w') as lineage_file:
                    json.dump(track_review_dict['tracks'], lineage_file, indent=1)
                    lineage_file.flush()
                    trks.add(lineage_file.name, 'lineage.json')

                with tempfile.NamedTemporaryFile() as raw_file:
                    np.save(raw_file, track_review_dict['X'])
                    raw_file.flush()
                    trks.add(raw_file.name, 'raw.npy')

                with tempfile.NamedTemporaryFile() as tracked_file:
                    np.save(tracked_file, track_review_dict['y_tracked'])
                    tracked_file.flush()
                    trks.add(tracked_file.name, 'tracked.npy')

        return track_review_dict

    def dump(self, filename):
        """Writes the state of the cell tracker to a .trk ('track') file.
        Includes raw & tracked images, and a lineage.json for parent/daughter
        information.
        """
        track_review_dict = self._track_review_dict()
        filename = pathlib.Path(filename)

        if filename.suffix != '.trk':
            filename = filename.with_suffix('.trk')

        filename = str(filename)

        with tarfile.open(filename, 'w') as trks:
            with tempfile.NamedTemporaryFile('w') as lineage_file:
                json.dump(track_review_dict['tracks'], lineage_file, indent=1)
                lineage_file.flush()
                trks.add(lineage_file.name, 'lineage.json')

            with tempfile.NamedTemporaryFile() as raw_file:
                np.save(raw_file, track_review_dict['X'])
                raw_file.flush()
                trks.add(raw_file.name, 'raw.npy')

            with tempfile.NamedTemporaryFile() as tracked_file:
                np.save(tracked_file, track_review_dict['y_tracked'])
                tracked_file.flush()
                trks.add(tracked_file.name, 'tracked.npy')

    def _track_to_graph(self, tracks):
        """Create a graph from the lineage information"""
        Dattr = {}
        edges = pd.DataFrame()

        for L in tracks.values():
            # Calculate node ids
            cellid = ['{}_{}'.format(L['label'], f) for f in L['frames']]
            # Add edges from cell ids
            edges = edges.append(pd.DataFrame({'source': cellid[0:-1],
                                               'target': cellid[1:]}))

            # Collect any division attributes
            if L['frame_div'] is not None:
                Dattr['{}_{}'.format(L['label'], L['frame_div'] - 1)] = {'division': True}

            # Create any daughter-parent edges
            if L['parent'] is not None:
                source = '{}_{}'.format(L['parent'], min(L['frames']) - 1)
                target = '{}_{}'.format(L['label'], min(L['frames']))
                edges = edges.append(pd.DataFrame({'source': [source],
                                                   'target': [target]}))

        G = nx.from_pandas_edgelist(edges, source='source', target='target')
        nx.set_node_attributes(G, Dattr)
        return G

    def _flag_false_pos(self, G, time_excl):
        """Examine graph for false positive nodes
        """

        # TODO: Current implementation may eliminate some divisions at the edge of the frame -
        #       Further research needed

        # Identify false positive nodes
        node_fix = []
        for g in nx.connected_component_subgraphs(G):
            div_nodes = [node for node, d in g.node.data() if d.get('division', False) is True]
            if len(div_nodes) > 1:
                for nd in div_nodes:
                    if g.degree(nd) == 2:
                        # Check how close suspected FP is to other known divisions
                        neighbors = list(G.neighbors(nd))

                        keep_div = True
                        for div_nd in div_nodes:
                            if div_nd != nd:
                                time_spacing = abs(int(nd.split('_')[1]) -
                                                   int(div_nd.split('_')[1]))
                                # If division is sufficiently far away
                                # we should exclude it from FP list
                                if time_spacing > time_excl:
                                    keep_div = False

                        if keep_div is True:
                            node_fix.append(nd)

        # Add supplementary information for each false positive
        D = {}
        for node in node_fix:
            D[node] = {
                'false positive': node,
                'neighbors': list(G.neighbors(node)),
                'connected lineages': set([int(n.split('_')[0])
                                          for n in nx.node_connected_component(G, n)])
            }

        return D

    def _review_candidate_nodes(self, FPs_candidates):
        """ review candidate false positive nodes and remove any errant degree 2 nodes.
        """
        FPs_presort = {}
        # review candidate false positive nodes and remove any errant degree 2 nodes
        for candidate_node in FPs_candidates:
            node = candidate_node[0]
            node_info = candidate_node[1]
            fp_label = int(node.split('_')[0])
            fp_frame = int(node.split('_')[1])

            neighbors = []  # structure will be [(neighbor1, frame), (neighbor2,frame)]
            for neighbor in node_info['neighbors']:
                neighbor_label = int(neighbor.split('_')[0])
                neighbor_frame = int(neighbor.split('_')[1])
                neighbors.append((neighbor_label, neighbor_frame))

            # if this cell only exists in one frame (and then it divides) but its 2 neighbors
            # both exist in the same frame it will be a degree 2 node but not be a false positive
            if neighbors[0][1] != neighbors[1][1]:
                FPs_presort[node] = node_info

        FPs_sorted = sorted(FPs_presort.items(), key=lambda v: int(v[0].split('_')[1]))

        return FPs_sorted

    def _remove_false_pos(self, lineage, tracked, FP_info):
        """ Remove nodes that have been identified as false positive divisions.
        """
        node = FP_info[0]
        node_info = FP_info[1]

        fp_label = int(node.split('_')[0])
        fp_frame = int(node.split('_')[1])

        neighbors = []  # structure will be [(neighbor1, frame), (neighbor2,frame)]
        for neighbor in node_info['neighbors']:
            neighbor_label = int(neighbor.split('_')[0])
            neighbor_frame = int(neighbor.split('_')[1])
            neighbors.append((neighbor_label, neighbor_frame))

        # Verify that the FP node only 2 neighbors - 1 before it and one after it
        if len(neighbors) == 2:
            # order the neighbors such that the time (frame order) is respected
            if neighbors[0][1] > neighbors[1][1]:
                temp = neighbors[0]
                neighbors[0] = neighbors[1]
                neighbors[1] = temp

            # Decide which labels to extend and which to remove

            # Neighbor_1 has same label as fp - the actual division hasnt occurred yet
            if fp_label == neighbors[0][0]:
                # The model mistakenly identified a division before the actual division occurred
                label_to_remove = neighbors[1][0]
                label_to_extend = neighbors[0][0]

                # Give all of the errant divisions information to the correct track
                lineage[label_to_extend]['frames'].extend(lineage[label_to_remove]['frames'])
                lineage[label_to_extend]['daughters'] = lineage[label_to_remove]['daughters']
                lineage[label_to_extend]['frame_div'] = lineage[label_to_remove]['frame_div']

                # Adjust the parent information for the actual daughters
                daughter_labels = lineage[label_to_remove]['daughters']
                for daughter in daughter_labels:
                    lineage[daughter]['parent'] = lineage[label_to_remove]['parent']

                # Remove the errant node from the annotated images
                channel = 0  # These images should only have one channel
                for frame in lineage[label_to_remove]['frames']:
                    label_loc = np.where(tracked[frame, :, :, channel] == label_to_remove)
                    tracked[frame, :, :, channel][label_loc] = label_to_extend

                # Remove the errant node from the lineage
                del lineage[label_to_remove]

            # Neighbor_2 has same label as fp - the actual division ocurred &
            # the model mistakenly allowed another
            # elif fp_label == neighbors[1][0]:
                # The model mistakenly identified a division after
                # the actual division occurred
                # label_to_remove = fp_label

            # Neither neighbor has same label as fp - the actual division
            # ocurred & the model mistakenly allowed another
            else:
                # The model mistakenly identified a division after the actual division occurred
                label_to_remove = fp_label
                label_to_extend = neighbors[1][0]

                # Give all of the errant divisions information to the correct track
                lineage[label_to_extend]['frames'] = \
                    lineage[fp_label]['frames'] + lineage[label_to_extend]['frames']
                lineage[label_to_extend]['parent'] = lineage[fp_label]['parent']

                # Adjust the parent information for the actual daughter
                parent_label = lineage[fp_label]['parent']
                for d_idx, daughter in enumerate(lineage[parent_label]['daughters']):
                    if daughter == fp_label:
                        lineage[parent_label]['daughters'][d_idx] = label_to_extend

                # Remove the errant node from the annotated images
                channel = 0  # These images should only have one channel
                for frame in lineage[label_to_remove]['frames']:
                    label_loc = np.where(tracked[frame, :, :, channel] == label_to_remove)
                    tracked[frame, :, :, channel][label_loc] = label_to_extend

                # Remove the errant node
                del lineage[label_to_remove]

        else:
            print('Error: More than 2 neighbor nodes')

        return lineage, tracked
