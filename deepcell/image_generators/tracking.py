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
"""Tracking data generators."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from skimage.measure import regionprops
from skimage.transform import resize

from tensorflow.keras import backend as K
from tensorflow.python.platform import tf_logging as logging
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SiameseDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Args:
        featurewise_center (bool): Set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center (bool): Set each sample mean to 0.
        featurewise_std_normalization (bool): Divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization (bool): Divide each input by its std.
        zca_epsilon (float): Epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening (bool): Apply ZCA whitening.
        rotation_range (int): Degree range for random rotations.
        width_shift_range (float): 1-D array-like or int

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              ``(-width_shift_range, +width_shift_range)``
            - With ``width_shift_range=2`` possible values are integers
              ``[-1, 0, +1]``, same as with ``width_shift_range=[-1, 0, +1]``,
              while with ``width_shift_range=1.0`` possible values are floats
              in the interval [-1.0, +1.0).

        height_shift_range: Float, 1-D array-like or int

            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              ``(-height_shift_range, +height_shift_range)``
            - With ``height_shift_range=2`` possible values
              are integers ``[-1, 0, +1]``,
              same as with ``height_shift_range=[-1, 0, +1]``,
              while with ``height_shift_range=1.0`` possible values are floats
              in the interval [-1.0, +1.0).

        shear_range (float): Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range (float): float or [lower, upper], Range for random zoom.
            If a float, ``[lower, upper] = [1-zoom_range, 1+zoom_range]``.
        channel_shift_range (float): range for random channel shifts.
        fill_mode (str): One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:

                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval (float): Value used for points outside the boundaries
            when ``fill_mode = "constant"``.
        horizontal_flip (bool): Randomly flip inputs horizontally.
        vertical_flip (bool): Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        validation_split (float): Fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def flow(self,
             train_dict,
             features,
             crop_dim=32,
             min_track_length=5,
             neighborhood_scale_size=64,
             neighborhood_true_size=100,
             sync_transform=True,
             batch_size=32,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return SiameseIterator(
            train_dict,
            self,
            features=features,
            crop_dim=crop_dim,
            min_track_length=min_track_length,
            neighborhood_scale_size=neighborhood_scale_size,
            neighborhood_true_size=neighborhood_true_size,
            sync_transform=sync_transform,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class SiameseIterator(Iterator):
    """Iterator yielding two sets of features (X) and the relationship (y).

    Features are passed in as a list of feature names, while the y is one of:
        {"same", "different", or "daughter"}.

    Args:
        train_dict (dict): Consists of numpy arrays for ``X`` and ``y``.
        image_data_generator (SiameseDataGenerator): For random transformations
            and normalization.
        features (list): Feature names to calculate and yield.
        crop_dim (int): Size of the resized appearance images
        min_track_length (int): Minimum number of frames to track over.
        neighborhood_scale_size (int): Size of resized neighborhood images
        neighborhood_true_size (int): Size of cropped neighborhood images
        sync_transform (bool): Whether to transform the features.
        batch_size (int): Size of a batch.
        shuffle (bool): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
        save_to_dir (str): Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix (str): Prefix to use for saving sample
            images (if ``save_to_dir`` is set).
        save_format (str): Format to use for saving sample images
            (if ``save_to_dir`` is set).
    """
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 features=['appearance', 'distance', 'neighborhood', 'regionprop'],
                 crop_dim=32,
                 min_track_length=5,
                 neighborhood_scale_size=64,
                 neighborhood_true_size=100,
                 sync_transform=True,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 3
            self.col_axis = 4
            self.time_axis = 2
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.time_axis = 1

        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = np.array(train_dict['y'], dtype='int32')

        if self.x.ndim != 5:
            raise ValueError('Input data in `SiameseIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)

        self.crop_dim = crop_dim
        self.min_track_length = min_track_length
        self.features = sorted(features)
        self.sync_transform = sync_transform
        self.neighborhood_scale_size = np.int(neighborhood_scale_size)
        self.neighborhood_true_size = np.int(neighborhood_true_size)
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.daughters = train_dict.get('daughters')
        if self.daughters is None:
            raise ValueError('`daughters` not found in `train_dict`. '
                             'Lineage information is required for training.')

        self._remove_bad_images()

        if self.x.shape[0] == 0:
            raise ValueError('Invalid data. No batch provided '
                             'with at least 2 cells.')

        self._create_track_ids()
        self._create_features()

        super(SiameseIterator, self).__init__(
            len(self.track_ids), batch_size, shuffle, seed)

    def _remove_bad_images(self):
        """Remove any batch of images that has fewer than 2 cell IDs."""
        good_batches = []
        for batch in range(self.x.shape[0]):
            # There should be at least 3 id's - 2 cells and 1 background
            if len(np.unique(self.y[batch])) > 2:
                good_batches.append(batch)

        X_new_shape = tuple([len(good_batches)] + list(self.x.shape)[1:])
        y_new_shape = tuple([len(good_batches)] + list(self.y.shape)[1:])

        X_new = np.zeros(X_new_shape, dtype=K.floatx())
        y_new = np.zeros(y_new_shape, dtype='int32')

        for k, batch in enumerate(good_batches):
            X_new[k] = self.x[batch]
            y_new[k] = self.y[batch]

        self.x = X_new
        self.y = y_new
        self.daughters = [self.daughters[i] for i in good_batches]

    def _create_track_ids(self):
        """Builds the track IDs.

        Creates unique cell IDs, as cell labels are NOT unique across batches.

        Returns:
            dict: A dict containing the batch and label of each each track.
        """
        track_counter = 0
        track_ids = {}
        for batch in range(self.y.shape[0]):
            y_batch = self.y[batch]
            daughters_batch = self.daughters[batch]
            num_cells = np.amax(y_batch)  # TODO: iterate over np.unique instead
            for cell in range(1, num_cells + 1):
                # count number of pixels cell occupies in each frame
                y_true = np.sum(y_batch == cell, axis=(self.row_axis - 1, self.col_axis - 1))

                # get indices of frames where cell is present
                if self.channel_axis == 1:
                    y_index = np.where(y_true > 0)[1]
                else:
                    y_index = np.where(y_true > 0)[0]

                if y_index.size > 3:  # if cell is present at all
                    # Only include daughters if there are enough frames in their tracks
                    if self.daughters is not None:
                        daughter_ids = daughters_batch.get(cell, [])
                        if daughter_ids:
                            daughter_track_lengths = []
                            for did in daughter_ids:
                                # Screen daughter tracks to make sure they are long enough
                                # Length currently set to 0
                                axis = (self.row_axis - 1, self.col_axis - 1)
                                d_true = np.sum(y_batch == did, axis=axis)
                                d_track_length = len(np.where(d_true > 0)[0])
                                daughter_track_lengths.append(d_track_length > 3)
                            keep_daughters = all(daughter_track_lengths)
                            daughters = daughter_ids if keep_daughters else []
                        else:
                            daughters = []
                    else:
                        daughters = []

                    track_ids[track_counter] = {
                        'batch': batch,
                        'label': cell,
                        'frames': y_index,
                        'daughters': daughters
                    }

                    track_counter += 1

                else:
                    y_batch[y_batch == cell] = 0

                    self.y[batch] = y_batch

        # Add a field to the track_ids dict that locates
        # all of the different cells in each frame
        for track in track_ids:
            track_ids[track]['different'] = {}
            batch = track_ids[track]['batch']
            cell_label = track_ids[track]['label']
            for frame in track_ids[track]['frames']:
                if self.channel_axis == 1:
                    y_unique = np.unique(self.y[batch, :, frame])
                else:
                    y_unique = np.unique(self.y[batch, frame])
                y_unique = np.delete(y_unique, np.where(y_unique == 0))
                y_unique = np.delete(y_unique, np.where(y_unique == cell_label))
                track_ids[track]['different'][frame] = y_unique

        # We will need to look up the track_ids of cells if we know their batch and label. We will
        # create a dictionary that stores this information
        reverse_track_ids = {}
        for batch in range(self.y.shape[0]):
            reverse_track_ids[batch] = {}

        for track in track_ids:
            batch = track_ids[track]['batch']
            cell_label = track_ids[track]['label']
            reverse_track_ids[batch][cell_label] = track

        # Save dictionaries
        self.track_ids = track_ids
        self.reverse_track_ids = reverse_track_ids

        # Identify which tracks have divisions
        self.tracks_with_divisions = []
        for track in self.track_ids:
            if self.track_ids[track]['daughters']:
                self.tracks_with_divisions.append(track)

    def _sub_area(self, X_frame, y_frame, cell_label, num_channels):
        if self.data_format == 'channels_first':
            X_frame = np.rollaxis(X_frame, 0, 3)
            y_frame = np.rollaxis(y_frame, 0, 3)

        pads = ((self.neighborhood_true_size, self.neighborhood_true_size),
                (self.neighborhood_true_size, self.neighborhood_true_size),
                (0, 0))

        X_padded = np.pad(X_frame, pads, mode='constant', constant_values=0)
        y_padded = np.pad(y_frame, pads, mode='constant', constant_values=0)
        props = regionprops(np.squeeze(np.int32(y_padded == cell_label)))
        center_x, center_y = props[0].centroid
        center_x, center_y = np.int(center_x), np.int(center_y)
        X_reduced = X_padded[
            center_x - self.neighborhood_true_size:center_x + self.neighborhood_true_size,
            center_y - self.neighborhood_true_size:center_y + self.neighborhood_true_size,
            :]

        # Resize X_reduced in case it is used instead of the neighborhood method
        resize_shape = (2 * self.neighborhood_scale_size + 1,
                        2 * self.neighborhood_scale_size + 1, num_channels)

        # Resize images from bounding box
        X_reduced = resize(X_reduced, resize_shape, mode='constant', preserve_range=True)

        if self.data_format == 'channels_first':
            X_reduced = np.rollaxis(X_reduced, -1, 0)

        return X_reduced

    def _get_features(self, X, y, frames, labels):
        """Gets the features of a list of cells.
           Cells are defined by lists of frames and labels.
           The i'th element of frames and labels is the frame and label of the
           i'th cell being grabbed.
        """
        channel_axis = self.channel_axis - 1
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

        if self.data_format == 'channels_first':
            neighborhood_shape = (len(frames),
                                  X.shape[channel_axis],
                                  2 * self.neighborhood_scale_size + 1,
                                  2 * self.neighborhood_scale_size + 1)
        else:
            neighborhood_shape = (len(frames),
                                  2 * self.neighborhood_scale_size + 1,
                                  2 * self.neighborhood_scale_size + 1,
                                  X.shape[channel_axis])

        # future area should not include last frame in movie
        last_frame = self.x.shape[self.time_axis] - 1
        if last_frame in frames:
            future_area_len = len(frames) - 1
        else:
            future_area_len = len(frames)

        future_area_shape = tuple([future_area_len] + list(neighborhood_shape)[1:])

        # Initialize storage for appearances and centroids
        appearances = np.zeros(appearance_shape, dtype=K.floatx())
        centroids = []
        rprops = []
        neighborhoods = np.zeros(neighborhood_shape, dtype=K.floatx())
        future_areas = np.zeros(future_area_shape, dtype=K.floatx())

        for counter, (frame, cell_label) in enumerate(zip(frames, labels)):
            # Get the bounding box
            X_frame = X[frame] if self.data_format == 'channels_last' else X[:, frame]
            y_frame = y[frame] if self.data_format == 'channels_last' else y[:, frame]

            props = regionprops(np.squeeze(np.int32(y_frame == cell_label)))
            minr, minc, maxr, maxc = props[0].bbox
            centroids.append(props[0].centroid)
            rprops.append(np.array([props[0].area, props[0].perimeter, props[0].eccentricity]))

            # Extract images from bounding boxes
            if self.data_format == 'channels_first':
                appearance = np.copy(X[:, frame, minr:maxr, minc:maxc])
                resize_shape = (X.shape[channel_axis], self.crop_dim, self.crop_dim)
            else:
                appearance = np.copy(X[frame, minr:maxr, minc:maxc, :])
                resize_shape = (self.crop_dim, self.crop_dim, X.shape[channel_axis])

            # Resize images from bounding box
            appearance = resize(appearance, resize_shape, mode='constant', preserve_range=True)
            if self.data_format == 'channels_first':
                appearances[:, counter] = appearance
            else:
                appearances[counter] = appearance

            neighborhoods[counter] = self._sub_area(
                X_frame, y_frame, cell_label, X.shape[channel_axis])

            if frame != last_frame:
                if self.data_format == 'channels_first':
                    X_future_frame = X[:, frame + 1]
                else:
                    X_future_frame = X[frame + 1]

                future_areas[counter] = self._sub_area(
                    X_future_frame, y_frame, cell_label, X.shape[channel_axis])

        return [appearances, centroids, neighborhoods, rprops, future_areas]

    def _create_features(self):
        """Gets the appearances of every cell, crops them out, resizes them,
        and stores them in an matrix. Pre-fetching the appearances should
        significantly speed up the generator. It also gets the centroids and
        neighborhoods.
        """
        number_of_tracks = len(self.track_ids.keys())

        # Initialize the array for the appearances and centroids
        if self.data_format == 'channels_first':
            all_appearances_shape = (number_of_tracks,
                                     self.x.shape[self.channel_axis],
                                     self.x.shape[self.time_axis],
                                     self.crop_dim,
                                     self.crop_dim)
        else:
            all_appearances_shape = (number_of_tracks,
                                     self.x.shape[self.time_axis],
                                     self.crop_dim,
                                     self.crop_dim,
                                     self.x.shape[self.channel_axis])
        all_appearances = np.zeros(all_appearances_shape, dtype=K.floatx())

        all_centroids_shape = (number_of_tracks, self.x.shape[self.time_axis], 2)
        all_centroids = np.zeros(all_centroids_shape, dtype=K.floatx())

        all_regionprops_shape = (number_of_tracks, self.x.shape[self.time_axis], 3)
        all_regionprops = np.zeros(all_regionprops_shape, dtype=K.floatx())

        if self.data_format == 'channels_first':
            all_neighborhoods_shape = (number_of_tracks,
                                       self.x.shape[self.channel_axis],
                                       self.x.shape[self.time_axis],
                                       2 * self.neighborhood_scale_size + 1,
                                       2 * self.neighborhood_scale_size + 1)
        else:
            all_neighborhoods_shape = (number_of_tracks,
                                       self.x.shape[self.time_axis],
                                       2 * self.neighborhood_scale_size + 1,
                                       2 * self.neighborhood_scale_size + 1,
                                       self.x.shape[self.channel_axis])

        all_neighborhoods = np.zeros(all_neighborhoods_shape, dtype=K.floatx())

        all_future_areas = np.zeros(all_neighborhoods_shape, dtype=K.floatx())

        for track in self.track_ids:
            batch = self.track_ids[track]['batch']
            cell_label = self.track_ids[track]['label']
            frames = self.track_ids[track]['frames']

            # Make an array of labels that the same length as the frames array
            labels = [cell_label] * len(frames)
            X = self.x[batch]
            y = self.y[batch]

            appearance, centroid, neighborhood, regionprop, future_area = self._get_features(
                X, y, frames, labels)

            if self.data_format == 'channels_first':
                all_appearances[track][:, np.array(frames), :, :] = appearance
            else:
                all_appearances[track, np.array(frames), :, :, :] = appearance

            all_centroids[track, np.array(frames), :] = centroid

            if self.data_format == 'channels_first':
                all_neighborhoods[track, :, np.array(frames), :] = neighborhood
            else:
                all_neighborhoods[track, np.array(frames), :, :] = neighborhood

            # future area should never include last frame
            last_frame = self.x.shape[self.time_axis] - 1
            if last_frame in frames:
                frames_without_last = [f for f in frames if f != last_frame]
                if self.data_format == 'channels_first':
                    all_future_areas[track, :, np.array(frames_without_last), :] = future_area
                else:
                    all_future_areas[track, np.array(frames_without_last), :, :] = future_area
            else:
                if self.data_format == 'channels_first':
                    all_future_areas[track, :, np.array(frames), :] = future_area
                else:
                    all_future_areas[track, np.array(frames), :, :] = future_area
            all_regionprops[track, np.array(frames), :] = regionprop

        self.all_appearances = all_appearances
        self.all_centroids = all_centroids
        self.all_regionprops = all_regionprops
        self.all_neighborhoods = all_neighborhoods
        self.all_future_areas = all_future_areas

    def _fetch_appearances(self, track, frames):
        """Gets the appearances after they have been cropped out of the image
        """
        # TODO: Check to make sure the frames are acceptable
        if self.data_format == 'channels_first':
            return self.all_appearances[track][:, np.array(frames), :, :]
        return self.all_appearances[track, np.array(frames), :, :, :]

    def _fetch_centroids(self, track, frames):
        """Gets the centroids after they have been extracted and stored
        """
        # TODO: Check to make sure the frames are acceptable
        return self.all_centroids[track, np.array(frames), :]

    def _fetch_neighborhoods(self, track, frames):
        """Gets the neighborhoods after they have been extracted and stored
        """
        # TODO: Check to make sure the frames are acceptable
        if self.data_format == 'channels_first':
            return self.all_neighborhoods[track][:, np.array(frames), :, :]
        return self.all_neighborhoods[track, np.array(frames), :, :, :]

    def _fetch_future_areas(self, track, frames):
        """Gets the future areas after they have been extracted and stored
        """
        # TODO: Check to make sure the frames are acceptable
        if self.data_format == 'channels_first':
            return self.all_future_areas[track][:, np.array(frames), :, :]
        return self.all_future_areas[track, np.array(frames), :, :, :]

    def _fetch_regionprops(self, track, frames):
        """Gets the regionprops after they have been extracted and stored
        """
        # TODO: Check to make sure the frames are acceptable
        return self.all_regionprops[track, np.array(frames)]

    def _fetch_frames(self, track, division=False):
        """Fetch a random interval of frames given a track:

        If division, grab the last ``min_track_length`` frames.
        Otherwise, grab any interval of frames of length
        ``min_track_length`` that does not include the
        last tracked frame.

        Args:
            track: integer, used to look up track ID
            division: boolean, is the event being tracked a division

        Returns:
            list: interval of frames of length ``min_track_length``
        """
        track_id = self.track_ids[track]

        # convert to list to use python's (+) on lists
        all_frames = list(track_id['frames'])

        if division:
            # sanity check
            if self.x.shape[self.time_axis] - 1 in all_frames:
                raise ValueError('Track {} is annotated incorrectly. '
                                 'No parent cell should be in the last frame '
                                 'of any movie.'.format(track_id))

            candidate_interval = all_frames[-self.min_track_length:]
        else:
            # exclude the final frame for comparison purposes
            candidate_frames = all_frames[:-1]

            # `start` is within [0, len(candidate_frames) - min_track_length]
            # in order to have at least `min_track_length` preceding frames.
            # The `max(..., 1)` is because `len(all_frames) <= self.min_track_length`
            # is possible. If `len(candidate_frames) <= self.min_track_length`,
            # then the interval will be the entire `candidate_frames`.
            high = max(len(candidate_frames) - self.min_track_length, 1)
            start = np.random.randint(0, high)
            candidate_interval = candidate_frames[start:start + self.min_track_length]

        # if the interval is too small, pad the interval with the oldest frame.
        if len(candidate_interval) < self.min_track_length:
            num_padding = self.min_track_length - len(candidate_interval)
            candidate_interval = [candidate_interval[0]] * num_padding + candidate_interval

        return candidate_interval

    def _compute_appearances(self, track_1, frames_1, track_2, frames_2, transform):
        appearance_1 = self._fetch_appearances(track_1, frames_1)
        appearance_2 = self._fetch_appearances(track_2, frames_2)

        # Apply random transforms
        new_appearance_1 = np.zeros(appearance_1.shape, dtype=K.floatx())
        new_appearance_2 = np.zeros(appearance_2.shape, dtype=K.floatx())

        for frame in range(appearance_1.shape[self.time_axis - 1]):
            if self.data_format == 'channels_first':
                if transform is not None:
                    app_temp = self.image_data_generator.apply_transform(
                        appearance_1[:, frame, :, :], transform)
                else:
                    app_temp = self.image_data_generator.random_transform(
                        appearance_1[:, frame, :, :])
                app_temp = self.image_data_generator.standardize(app_temp)
                new_appearance_1[:, frame, :, :] = app_temp

            if self.data_format == 'channels_last':
                if transform is not None:
                    app_temp = self.image_data_generator.apply_transform(
                        appearance_1[frame], transform)
                else:
                    self.image_data_generator.random_transform(appearance_1[frame])
                app_temp = self.image_data_generator.standardize(app_temp)
                new_appearance_1[frame] = app_temp

        if self.data_format == 'channels_first':
            if transform is not None:
                app_temp = self.image_data_generator.apply_transform(
                    appearance_2[:, 0, :, :], transform)
            else:
                app_temp = self.image_data_generator.random_transform(
                    appearance_2[:, 0, :, :])
            app_temp = self.image_data_generator.standardize(app_temp)
            new_appearance_2[:, 0, :, :] = app_temp

        if self.data_format == 'channels_last':
            if transform is not None:
                app_temp = self.image_data_generator.apply_transform(
                    appearance_2[0], transform)
            else:
                app_temp = self.image_data_generator.random_transform(appearance_2[0])
            app_temp = self.image_data_generator.standardize(app_temp)
            new_appearance_2[0] = app_temp

        return new_appearance_1, new_appearance_2

    def _compute_distances(self, track_1, frames_1, track_2, frames_2, transform):
        centroid_1 = self._fetch_centroids(track_1, frames_1)
        centroid_2 = self._fetch_centroids(track_2, frames_2)

        # Compute distances between centroids
        centroids = np.concatenate([centroid_1, centroid_2], axis=0)
        distance = np.diff(centroids, axis=0)
        zero_pad = np.zeros((1, 2), dtype=K.floatx())
        distance = np.concatenate([zero_pad, distance], axis=0)

        # Randomly rotate and expand all the distances
        # TODO(enricozb): Investigate effect of rotations, it should be invariant

        distance_1 = distance[0:-1, :]
        distance_2 = distance[-1, :]

        return distance_1, distance_2

    def _compute_regionprops(self, track_1, frames_1, track_2, frames_2, transform):
        regionprop_1 = self._fetch_regionprops(track_1, frames_1)
        regionprop_2 = self._fetch_regionprops(track_2, frames_2)
        return regionprop_1, regionprop_2

    def _compute_neighborhoods(self, track_1, frames_1, track_2, frames_2, transform):
        track_2, frames_2 = None, None  # To guarantee we don't use these.
        neighborhood_1 = self._fetch_neighborhoods(track_1, frames_1)
        neighborhood_2 = self._fetch_future_areas(track_1, [frames_1[-1]])

        axis = 1 if self.data_format == 'channels_first' else 0

        neighborhoods = np.concatenate([neighborhood_1, neighborhood_2], axis=axis)

        for frame in range(neighborhoods.shape[self.time_axis - 1]):
            if self.data_format == 'channels_first':
                neigh_temp = neighborhoods[:, frame]
            else:
                neigh_temp = neighborhoods[frame]
            if transform is not None:
                neigh_temp = self.image_data_generator.apply_transform(
                    neigh_temp, transform)
            else:
                neigh_temp = self.image_data_generator.random_transform(neigh_temp)

            if self.data_format == 'channels_first':
                neighborhoods[:, frame] = neigh_temp
            else:
                neighborhoods[frame] = neigh_temp

        if self.data_format == 'channels_first':
            neighborhood_1 = neighborhoods[:, 0:-1, :, :]
            neighborhood_2 = neighborhoods[:, -1:, :, :]
        else:
            neighborhood_1 = neighborhoods[0:-1, :, :, :]
            neighborhood_2 = neighborhoods[-1:, :, :, :]

        return neighborhood_1, neighborhood_2

    def _compute_feature_shape(self, feature, index_array):
        if feature == 'appearance':
            if self.data_format == 'channels_first':
                shape_1 = (len(index_array),
                           self.x.shape[self.channel_axis],
                           self.min_track_length,
                           self.crop_dim,
                           self.crop_dim)
                shape_2 = (len(index_array),
                           self.x.shape[self.channel_axis],
                           1,
                           self.crop_dim,
                           self.crop_dim)
            else:
                shape_1 = (len(index_array),
                           self.min_track_length,
                           self.crop_dim,
                           self.crop_dim,
                           self.x.shape[self.channel_axis])
                shape_2 = (len(index_array),
                           1,
                           self.crop_dim,
                           self.crop_dim,
                           self.x.shape[self.channel_axis])

        elif feature == 'distance':
            shape_1 = (len(index_array), self.min_track_length, 2)
            shape_2 = (len(index_array), 1, 2)

        elif feature == 'neighborhood':
            if self.data_format == 'channels_first':
                shape_1 = (len(index_array),
                           self.x.shape[self.channel_axis],
                           self.min_track_length,
                           2 * self.neighborhood_scale_size + 1,
                           2 * self.neighborhood_scale_size + 1)
                shape_2 = (len(index_array),
                           self.x.shape[self.channel_axis],
                           1,
                           2 * self.neighborhood_scale_size + 1,
                           2 * self.neighborhood_scale_size + 1)
            else:
                shape_1 = (len(index_array),
                           self.min_track_length,
                           2 * self.neighborhood_scale_size + 1,
                           2 * self.neighborhood_scale_size + 1,
                           self.x.shape[self.channel_axis])
                shape_2 = (len(index_array),
                           1,
                           2 * self.neighborhood_scale_size + 1,
                           2 * self.neighborhood_scale_size + 1,
                           self.x.shape[self.channel_axis])
        elif feature == 'regionprop':
            shape_1 = (len(index_array), self.min_track_length, 3)
            shape_2 = (len(index_array), 1, 3)
        else:
            raise ValueError('_compute_feature_shape: '
                             'Unknown feature `{}`'.format(feature))

        return shape_1, shape_2

    def _compute_feature(self, feature, *args, **kwargs):
        if feature == 'appearance':
            return self._compute_appearances(*args, **kwargs)
        elif feature == 'distance':
            return self._compute_distances(*args, **kwargs)
        elif feature == 'neighborhood':
            return self._compute_neighborhoods(*args, **kwargs)
        elif feature == 'regionprop':
            return self._compute_regionprops(*args, **kwargs)
        else:
            raise ValueError('_compute_feature: '
                             'Unknown feature `{}`'.format(feature))

    def _get_batches_of_transformed_samples(self, index_array):
        # Initialize batch_x_1, batch_x_2, and batch_y, and cell distance
        # Compare cells in neighboring frames.
        # Select a sequence of cells/distances for x1 and 1 cell/distance for x2

        # setup zeroed batch arrays for each feature & batch_y
        batch_features = []
        for feature in self.features:
            shape_1, shape_2 = self._compute_feature_shape(feature, index_array)
            batch_features.append([np.zeros(shape_1, dtype=K.floatx()),
                                   np.zeros(shape_2, dtype=K.floatx())])

        batch_y = np.zeros((len(index_array), 3), dtype='int32')

        for i, j in enumerate(index_array):
            # Identify which tracks are going to be selected
            track_id = self.track_ids[j]
            batch = track_id['batch']
            label_1 = track_id['label']

            # Choose comparison cell
            # Determine what class the track will be - different (0), same (1), division (2)
            division = False
            type_cell = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])

            # Dealing with edge cases
            # If class is division, check if the first cell divides. If not, change tracks
            if type_cell == 2:
                division = True
                if not track_id['daughters']:
                    # No divisions so randomly choose a different track that is
                    # guaranteed to have a division
                    new_j = np.random.choice(self.tracks_with_divisions)
                    j = new_j
                    track_id = self.track_ids[j]
                    batch = track_id['batch']
                    label_1 = track_id['label']

            # Get the frames for cell 1 and frames/label for cell 2
            frames_1 = self._fetch_frames(j, division=division)
            if len(frames_1) != self.min_track_length:
                logging.warning('self._fetch_frames(%s, division=%s) returned'
                                ' %s frames but %s frames were expected.',
                                j, division, len(frames_1),
                                self.min_track_length)

            # If the type is not 2 (not division) then the last frame is not
            # included in `frames_1`, so we grab that to use for comparison
            if type_cell != 2:
                # For frame_2, choose the next frame cell 1 appears in
                last_frame_1 = np.amax(frames_1)

                # logging.warning('last_frame_1: %s', last_frame_1)
                # logging.warning('track_id_frames: %s', track_id['frames'])

                frame_2 = np.amin([x for x in track_id['frames'] if x > last_frame_1])
                frames_2 = [frame_2]

                different_cells = track_id['different'][frame_2]

            if type_cell == 0:
                # If there are no different cells in the subsequent frame, we must choose
                # the same cell
                if len(different_cells) == 0:
                    type_cell = 1
                else:
                    label_2 = np.random.choice(different_cells)

            if type_cell == 1:
                # If there is only 1 cell in frame_2, we can only choose the class to be same
                label_2 = label_1

            if type_cell == 2:
                # There should always be 2 daughters but not always a valid label
                label_2 = np.int(np.random.choice(track_id['daughters']))
                daughter_track = self.reverse_track_ids[batch][label_2]
                frame_2 = np.amin(self.track_ids[daughter_track]['frames'])
                frames_2 = [frame_2]

            track_1 = j
            track_2 = self.reverse_track_ids[batch][label_2]

            # compute desired features & save them to the batch arrays
            if self.sync_transform:
                # random angle & flips
                flip_h = self.image_data_generator.horizontal_flip
                flip_v = self.image_data_generator.vertical_flip
                transform = {
                    'theta': self.image_data_generator.rotation_range * np.random.uniform(-1, 1),
                    'flip_horizontal': (np.random.random() < 0.5) if flip_h else False,
                    'flip_vertical': (np.random.random() < 0.5) if flip_v else False
                }

            else:
                transform = None

            for feature_i, feature in enumerate(self.features):
                feature_1, feature_2 = self._compute_feature(
                    feature, track_1, frames_1, track_2, frames_2, transform=transform)

                batch_features[feature_i][0][i] = feature_1
                batch_features[feature_i][1][i] = feature_2

            batch_y[i, type_cell] = 1

        # create dictionary to house generator outputs
        # Start with batch inputs to model
        batch_inputs = {}
        for feature_i, feature in enumerate(self.features):
            batch_feature_1, batch_feature_2 = batch_features[feature_i]
            # Remove singleton dimensions (if min_track_length is 1)
            if self.min_track_length < 2:
                axis = self.time_axis if feature == 'appearance' else 1
                batch_feature_1 = np.squeeze(batch_feature_1, axis=axis)
                batch_feature_2 = np.squeeze(batch_feature_2, axis=axis)

            batch_inputs['{}_input1'.format(feature)] = batch_feature_1
            batch_inputs['{}_input2'.format(feature)] = batch_feature_2

        # Dict to house training output (model target)
        batch_outputs = {'classification': batch_y}

        return batch_inputs, batch_outputs

    def next(self):
        """For python 2.x. Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
