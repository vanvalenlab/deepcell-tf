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
"""Semantic segmentation data generators."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

import numpy as np
from skimage.transform import rescale, resize

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.platform import tf_logging as logging

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None

from deepcell.image_generators import _transform_masks


class SemanticIterator(Iterator):
    """Iterator yielding data from Numpy arrays (``X`` and ``y``).

    Args:
        train_dict (dict): Consists of numpy arrays for ``X`` and ``y``.
        image_data_generator (ImageDataGenerator): For random transformations
            and normalization.
        batch_size (int): Size of a batch.
        min_objects (int): Images with fewer than ``min_objects`` are ignored.
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
                 batch_size=1,
                 shuffle=False,
                 transforms=['outer-distance'],
                 transforms_kwargs={},
                 seed=None,
                 min_objects=3,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        # Load data
        if 'X' not in train_dict:
            raise ValueError('No training data found in train_dict')

        if 'y' not in train_dict:
            raise ValueError('Instance masks are required for the '
                             'SemanticIterator')

        X, y = train_dict['X'], train_dict['y']

        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))

        if X.ndim != 4:
            raise ValueError('Input data in `SemanticIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', X.shape)

        self.x = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype='int32')

        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.min_objects = min_objects

        # Remove images with small numbers of cells
        invalid_batches = []
        for b in range(self.x.shape[0]):
            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.x = np.delete(self.x, invalid_batches, axis=0)
        self.y = np.delete(self.y, invalid_batches, axis=0)

        super(SemanticIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):

            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            for transform in self.transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=self.data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        return y_semantic_list

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]),
                           dtype=self.x.dtype)
        batch_y = []

        for i, j in enumerate(index_array):
            x = self.x[j]

            # _transform_labels expects batch dimension
            y_semantic_list = self._transform_labels(self.y[j:j + 1])

            # initialize batch_y
            if len(batch_y) == 0:
                for ys in y_semantic_list:
                    shape = tuple([len(index_array)] + list(ys.shape[1:]))
                    batch_y.append(np.zeros(shape, dtype=ys.dtype))

            # random_transform does not expect batch dimension
            y_semantic_list = [ys[0] for ys in y_semantic_list]

            # Apply transformation
            x, y_semantic_list = self.image_data_generator.random_transform(
                x, y_semantic_list)

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

            for k, ys in enumerate(y_semantic_list):
                batch_y[k][i] = ys

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                if self.data_format == 'channels_first':
                    img_x = np.expand_dims(batch_x[i, 0, ...], 0)
                else:
                    img_x = np.expand_dims(batch_x[i, ..., 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.y is not None:
                    # Save argmax of y batch
                    for k, y_sem in enumerate(batch_y):
                        if y_sem[i].shape[self.channel_axis - 1] == 1:
                            img_y = y_sem[i]
                        else:
                            img_y = np.argmax(y_sem[i],
                                              axis=self.channel_axis - 1)
                            img_y = np.expand_dims(img_y,
                                                   axis=self.channel_axis - 1)
                        img = array_to_img(img_y, self.data_format, scale=True)
                        fname = 'y_{sem}_{prefix}_{index}_{hash}.{format}'.format(
                            sem=k,
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                        img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y

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


class SemanticDataGenerator(ImageDataGenerator):
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
             batch_size=1,
             transforms=['outer-distance'],
             transforms_kwargs={},
             min_objects=3,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict (dict): Consists of numpy arrays for ``X`` and ``y``.
            batch_size (int): Size of a batch. Defaults to 1.
            shuffle (bool): Whether to shuffle the data between epochs.
                Defaults to ``True``.
            seed (int): Random seed for data shuffling.
            min_objects (int): Minumum number of objects allowed per image
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if ``save_to_dir`` is set).
            save_format (str): Format to use for saving sample images
                (if ``save_to_dir`` is set).

        Returns:
            SemanticIterator: An ``Iterator`` yielding tuples of ``(x, y)``,
            where ``x`` is a numpy array of image data and ``y`` is list of
            numpy arrays of transformed masks of the same shape.
        """
        return SemanticIterator(
            train_dict,
            self,
            batch_size=batch_size,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            shuffle=shuffle,
            min_objects=min_objects,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x (numpy.array): 3D tensor or list of 3D tensors,
                single image.
            y (numpy.array): 3D tensor or list of 3D tensors,
                label mask(s) for ``x``, optional.
            seed (int): Random seed.

        Returns:
            numpy.array: A randomly transformed copy of the input (same shape).
            If ``y`` is passed, it is transformed if necessary and returned.
        """
        params = self.get_random_transform(x.shape, seed)

        if isinstance(x, list):
            x = [self.apply_transform(x_i, params) for x_i in x]
        else:
            x = self.apply_transform(x, params)

        if y is None:
            return x

        # Nullify the transforms that don't affect `y`
        params['brightness'] = None
        params['channel_shift_intensity'] = None
        _interpolation_order = self.interpolation_order
        self.interpolation_order = 0

        if isinstance(y, list):
            y_new = []
            for y_i in y:
                if y_i.shape[self.channel_axis - 1] > 1:
                    y_t = self.apply_transform(y_i, params)

                # Keep original interpolation order if it is a
                # regression task
                elif y_i.shape[self.channel_axis - 1] == 1:
                    self.interpolation_order = _interpolation_order
                    y_t = self.apply_transform(y_i, params)
                    self.interpolation_order = 0
                y_new.append(y_t)
            y = y_new
        else:
            y = self.apply_transform(y, params)

        self.interpolation_order = _interpolation_order
        return x, y


class SemanticMovieIterator(Iterator):
    """Iterator yielding data from Numpy arrays (``X`` and ``y``).

    Args:
        train_dict (dict): Dictionary consisting of numpy arrays
            for ``X`` and ``y``.
        movie_data_generator (SemanticMovieGenerator): ``SemanticMovieGenerator``
            to use for random transformations and normalization.
        batch_size (int): Size of a batch.
        frames_per_batch (int): Size of z-axis in generated batches.
        shuffle (boolean): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        min_objects (int): Minumum number of objects allowed per image.
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
                 movie_data_generator,
                 batch_size=1,
                 frames_per_batch=5,
                 shuffle=False,
                 transforms=['outer-distance'],
                 transforms_kwargs={},
                 seed=None,
                 min_objects=3,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        # Load data
        if 'X' not in train_dict:
            raise ValueError('No training data found in train_dict')

        if 'y' not in train_dict:
            raise ValueError('Instance masks are required for the '
                             'SemanticMovieIterator')

        X, y = train_dict['X'], train_dict['y']

        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))

        if X.ndim != 5:
            raise ValueError('Input data in `SemanticMovieIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', X.shape)

        self.x = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype='int32')
        self.frames_per_batch = frames_per_batch
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.row_axis = 2 if data_format == 'channels_last' else 3
        self.col_axis = 3 if data_format == 'channels_last' else 4
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.min_objects = min_objects
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        if X.shape[self.time_axis] - frames_per_batch < 0:
            raise ValueError(
                'The number of frames used in each training batch should '
                'be less than the number of frames in the training data!')

        # Remove images with small numbers of cells
        invalid_batches = []
        for b in range(self.x.shape[0]):
            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.x = np.delete(self.x, invalid_batches, axis=0)
        self.y = np.delete(self.y, invalid_batches, axis=0)

        super(SemanticMovieIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _transform_labels(self, y):
        y_semantic_list = []

        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):

            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            for transform in self.transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=self.data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        return y_semantic_list

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            shape = (len(index_array), self.x.shape[1], self.frames_per_batch,
                     self.x.shape[3], self.x.shape[4])
        else:
            shape = tuple([len(index_array), self.frames_per_batch] +
                          list(self.x.shape)[2:])

        batch_x = np.zeros(shape, dtype=self.x.dtype)
        batch_y = []

        for i, j in enumerate(index_array):
            last_frame = self.x.shape[self.time_axis] - self.frames_per_batch
            time_start = np.random.randint(0, high=last_frame)
            time_end = time_start + self.frames_per_batch

            if self.time_axis == 1:
                x = self.x[j, time_start:time_end, ...]
                y = self.y[j:j + 1, time_start:time_end, ...]
            else:
                x = self.x[j, :, time_start:time_end, ...]
                y = self.y[j:j + 1, :, time_start:time_end, ...]

            # _transform_labels expects batch dimension
            y_semantic_list = self._transform_labels(y)

            # initialize batch_y
            if len(batch_y) == 0:
                for ys in y_semantic_list:
                    shape = tuple([len(index_array)] + list(ys.shape[1:]))
                    batch_y.append(np.zeros(shape, dtype=ys.dtype))

            # random_transform does not expect batch dimension
            y_semantic_list = [ys[0] for ys in y_semantic_list]

            # Apply transformation
            x, y_semantic_list = self.movie_data_generator.random_transform(
                x, y_semantic_list)

            x = self.movie_data_generator.standardize(x)

            batch_x[i] = x

            for k, ys in enumerate(y_semantic_list):
                batch_y[k][i] = ys

        if self.save_to_dir:
            time_axis = 2 if self.data_format == 'channels_first' else 1
            for i, j in enumerate(index_array):
                for frame in range(batch_x.shape[time_axis]):
                    if time_axis == 2:
                        img = array_to_img(batch_x[i, :, frame],
                                           self.data_format, scale=True)
                    else:
                        img = array_to_img(batch_x[i, frame],
                                           self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

                    if self.y is not None:
                        # Save argmax of y batch
                        if self.time_axis == 2:
                            img_y = np.argmax(batch_y[0][i, :, frame],
                                              axis=0)
                            img_channel_axis = 0
                            img_y = batch_y[0][i, :, frame]
                        else:
                            img_channel_axis = -1
                            img_y = batch_y[0][i, frame]
                        img_y = np.argmax(img_y, axis=img_channel_axis)
                        img_y = np.expand_dims(img_y, axis=img_channel_axis)
                        img = array_to_img(img_y, self.data_format, scale=True)
                        fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                        img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y

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


class SemanticMovieGenerator(ImageDataGenerator):
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
    def __init__(self, **kwargs):
        super(SemanticMovieGenerator, self).__init__(**kwargs)
        # Change the axes for 5D data
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 3
            self.col_axis = 4
            self.time_axis = 2
        if self.data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.time_axis = 1

    def flow(self,
             train_dict,
             batch_size=1,
             frames_per_batch=5,
             transforms=['outer-distance'],
             transforms_kwargs={},
             shuffle=True,
             min_objects=3,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict (dict): Consists of numpy arrays for ``X`` and ``y``.
            batch_size (int): Size of a batch.
            frames_per_batch (int): Size of z axis in generated batches.
            shuffle (bool): Whether to shuffle the data between epochs.
            seed (int): Random seed for data shuffling.
            min_objects (int): Images with fewer than ``min_objects``
                are ignored.
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if ``save_to_dir`` is set).
            save_format (str): Format to use for saving sample images
                (if ``save_to_dir`` is set).

        Returns:
            SemanticMovieIterator: An ``Iterator`` yielding tuples of
            ``(x, y)``, where ``x`` is a numpy array of image data and
            ``y`` is list of numpy arrays of transformed masks of the
            same shape.
        """
        return SemanticMovieIterator(
            train_dict,
            self,
            batch_size=batch_size,
            frames_per_batch=frames_per_batch,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            shuffle=shuffle,
            min_objects=min_objects,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        Args:
            x (numpy.array): batch of inputs to be normalized.

        Returns:
            numpy.array: The normalized inputs.
        """
        # TODO: standardize each image, not all frames at once
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) +
                  K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it '
                                'hasn\'t been fit on any training data. '
                                'Fit it first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it '
                                'hasn\'t been fit on any training data. Fit '
                                'it first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`zca_whitening`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
        return x

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        Args:
            x (numpy.array): The data to fit on. Should have rank 5.
            augment (bool): Whether to fit on randomly augmented samples.
            rounds (bool): If augment,
                how many augmentation passes to do over the data.
            seed (int): Random seed for data shuffling.

        Raises:
            ValueError: If input rank is not 5.
            ImportError: If zca_whitening is used and scipy is not available.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            logging.warning(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' +
                self.data_format + '" (channels on axis ' +
                str(self.channel_axis) + '), i.e. expected '
                'either 1, 3 or 4 channels on axis ' +
                str(self.channel_axis) + '. '
                'However, it was passed an array with shape ' +
                str(x.shape) + ' (' + str(x.shape[self.channel_axis]) +
                ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype)
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            axis = (0, self.time_axis, self.row_axis, self.col_axis)
            self.mean = np.mean(x, axis=axis)
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            axis = (0, self.time_axis, self.row_axis, self.col_axis)
            self.std = np.std(x, axis=axis)
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            if scipy is None:
                raise ImportError('Using zca_whitening requires SciPy. '
                                  'Install SciPy.')
            flat_x = np.reshape(
                x, (x.shape[0],
                    x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = scipy.linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x (numpy.array): 4D tensor or list of 4D tensors.
            y (numpy.array): 4D tensor or list of 4D tensors,
                label mask(s) for x, optional.
            seed (int): Random seed.

        Returns:
            numpy.array: A randomly transformed copy of the input (same shape).
            If ``y`` is passed, it is transformed if necessary and returned.
        """
        self.row_axis -= 1
        self.col_axis -= 1
        self.time_axis -= 1
        self.channel_axis -= 1

        x = x if isinstance(x, list) else [x]
        params = self.get_random_transform(x[0].shape, seed)

        for i in range(len(x)):
            x_i = x[i]
            for frame in range(x_i.shape[self.time_axis]):
                if self.data_format == 'channels_first':
                    x_trans = self.apply_transform(x_i[:, frame], params)
                    x_i[:, frame] = np.rollaxis(x_trans, -1, 0)
                else:
                    x_i[frame] = self.apply_transform(x_i[frame], params)
            x[i] = x_i

        x = x[0] if len(x) == 1 else x

        if y is not None:
            params['brightness'] = None
            params['channel_shift_intensity'] = None

            _interpolation_order = self.interpolation_order

            y = y if isinstance(y, list) else [y]

            for i in range(len(y)):
                y_i = y[i]

                order = 0 if y_i.shape[self.channel_axis] > 1 else _interpolation_order
                self.interpolation_order = order

                for frame in range(y_i.shape[self.time_axis]):
                    if self.data_format == 'channels_first':
                        y_trans = self.apply_transform(y_i[:, frame], params)
                        y_i[:, frame] = np.rollaxis(y_trans, 1, 0)
                    else:
                        y_i[frame] = self.apply_transform(y_i[frame], params)

                y[i] = y_i

                self.interpolation_order = _interpolation_order

            y = y[0] if len(y) == 1 else y

        # Note: Undo workaround
        self.row_axis += 1
        self.col_axis += 1
        self.time_axis += 1
        self.channel_axis += 1

        if y is None:
            return x

        return x, y


class Semantic3DIterator(Iterator):
    """Iterator yielding data from Numpy arrays (X and y).

    Args:
        train_dict (dict): Dictionary consisting of numpy arrays for ``X`` and ``y``.
        3d_data_generator (Semantic3DGenerator): ``Semantic3DGenerator``
            to use for random transformations and normalization.
        batch_size (int): Size of a batch.
        frames_per_batch (int): Size of z-axis in generated batches.
        frame_shape (tuple): Shape of the cropped frames.
        shuffle (bool): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        min_objects (int): Minumum number of objects allowed per image.
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
                 data_generator_3d,
                 batch_size=1,
                 frames_per_batch=5,
                 frame_shape=None,
                 shuffle=False,
                 transforms=['outer-distance'],
                 transforms_kwargs={},
                 aug_3d=False,
                 rotation_3d=0,
                 sampling=None,
                 z_scale=None,
                 seed=None,
                 min_objects=3,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):

        # Load data
        if 'X' not in train_dict:
            raise ValueError('No training data found in train_dict')

        if 'y' not in train_dict:
            raise ValueError('Instance masks are required for the '
                             'Semantic3DIterator')

        X, y = train_dict['X'], train_dict['y']

        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))

        if X.ndim != 5:
            raise ValueError('Input data in `Semantic3DIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', X.shape)

        if rotation_3d > 0 and not z_scale:
            raise ValueError('z_scaling factor required to rotate in 3d')

        def _scale_im(input_im, scale, order):

            dtype = input_im.dtype
            batch_list = []
            for batch_num in range(input_im.shape[0]):
                batch = input_im[batch_num, ...]

                if data_format == 'channels_first':

                    batch = np.moveaxis(batch, 0, -1)
                    rescaled = rescale(batch, scale,
                                       order=order,
                                       preserve_range=True,
                                       multichannel=True)
                    rescaled = np.moveaxis(rescaled, -1, 0)

                else:
                    rescaled = rescale(batch, scale,
                                       order=order,
                                       preserve_range=True,
                                       multichannel=True)

                batch_list.append(rescaled)
            return np.stack(batch_list, axis=0).astype(dtype)

        if aug_3d and rotation_3d > 0:
            scale = tuple([z_scale, 1, 1])
            X = _scale_im(X, scale, order=1)
            y = _scale_im(y, scale, order=0)

            self.output_frames = frames_per_batch
            frames_per_batch = int(round(frames_per_batch * z_scale))

        self.x = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype='int32')
        self.frames_per_batch = frames_per_batch
        self.frame_shape = frame_shape
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.aug_3d = aug_3d  # TODO: Add documentation
        self.rotation_3d = rotation_3d  # TODO: Add documentation
        self.z_scale = z_scale  # TODO: Add documentation
        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.row_axis = 2 if data_format == 'channels_last' else 3
        self.col_axis = 3 if data_format == 'channels_last' else 4
        self.data_generator_3d = data_generator_3d
        self.data_format = data_format
        self.min_objects = min_objects
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        if X.shape[self.time_axis] - frames_per_batch < 0:
            raise ValueError(
                'The number of frames used in each training batch should '
                'be less than the number of frames in the training data!'
                'fpb is {} and timeaxis is {}'.format(frames_per_batch, X.shape[self.time_axis]))

        invalid_batches = []

        # Remove images with small numbers of cells
        # TODO: make this work with the cropping implementation
        for b in range(self.x.shape[0]):
            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.x = np.delete(self.x, invalid_batches, axis=0)
        self.y = np.delete(self.y, invalid_batches, axis=0)

        super(Semantic3DIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _transform_labels(self, y):
        y_semantic_list = []
        # loop over channels axis of labels in case there are multiple label types
        for label_num in range(y.shape[self.channel_axis]):

            if self.channel_axis == 1:
                y_current = y[:, label_num:label_num + 1, ...]
            else:
                y_current = y[..., label_num:label_num + 1]

            for transform in self.transforms:
                transform_kwargs = self.transforms_kwargs.get(transform, dict())
                y_transform = _transform_masks(y_current, transform,
                                               data_format=self.data_format,
                                               **transform_kwargs)
                y_semantic_list.append(y_transform)

        return y_semantic_list

    def _get_batches_of_transformed_samples(self, index_array):
        if self.frame_shape:
            rows = self.frame_shape[0]
            cols = self.frame_shape[1]
        else:
            rows = self.x.shape[self.row_axis]
            cols = self.x.shape[self.col_axis]

        if self.data_format == 'channels_first':
            shape = (len(index_array), self.x.shape[1], self.frames_per_batch,
                     rows, cols)
        else:
            shape = (len(index_array), self.frames_per_batch,
                     rows, cols, self.x.shape[4])
        batch_x = np.zeros(shape, dtype=self.x.dtype)
        batch_y = []

        for i, j in enumerate(index_array):
            last_frame = self.x.shape[self.time_axis] - self.frames_per_batch
            if last_frame == 0:
                time_start = 0
            else:
                time_start = np.random.randint(0, high=last_frame)

            time_end = time_start + self.frames_per_batch

            if self.frame_shape:
                last_row = self.x.shape[self.row_axis] - self.frame_shape[0]
                last_col = self.x.shape[self.col_axis] - self.frame_shape[1]

                row_start = 0 if last_row == 0 else np.random.randint(0, high=last_row)
                col_start = 0 if last_col == 0 else np.random.randint(0, high=last_col)

                row_end = row_start + self.frame_shape[0]
                col_end = col_start + self.frame_shape[1]
            else:
                row_start, row_end = 0, self.x.shape[self.row_axis]
                col_start, col_end = 0, self.x.shape[self.col_axis]

            if self.time_axis == 1:
                x = self.x[j, time_start:time_end, row_start:row_end, col_start:col_end, :]
                y = self.y[j:j + 1, time_start:time_end, row_start:row_end, col_start:col_end, :]
            else:
                x = self.x[j, :, time_start:time_end, row_start:row_end, col_start:col_end]
                y = self.y[j:j + 1, :, time_start:time_end, row_start:row_end, col_start:col_end]

            # _transform_labels expects batch dimension
            y_semantic_list = self._transform_labels(y)

            # initialize batch_y
            if len(batch_y) == 0:
                for ys in y_semantic_list:
                    shape = tuple([len(index_array)] + list(ys.shape[1:]))
                    batch_y.append(np.zeros(shape, dtype=ys.dtype))

            # random_transform does not expect batch dimension
            y_semantic_list = [ys[0] for ys in y_semantic_list]

            # Apply transformation
            x, y_semantic_list = self.data_generator_3d.random_transform(
                x, y_semantic_list,
                aug_3d=self.aug_3d,
                rotation_3d=self.rotation_3d)

            x = self.data_generator_3d.standardize(x)

            batch_x[i] = x

            for k, y_sem in enumerate(y_semantic_list):
                batch_y[k][i] = y_sem

            if self.save_to_dir:
                time_axis = 2 if self.data_format == 'channels_first' else 1
                for i, j in enumerate(index_array):
                    for frame in range(batch_x.shape[time_axis]):
                        if time_axis == 2:
                            img = array_to_img(batch_x[i, :, frame],
                                               self.data_format, scale=True)
                        else:
                            img = array_to_img(batch_x[i, frame],
                                               self.data_format, scale=True)
                        fname = '{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                        img.save(os.path.join(self.save_to_dir, fname))

                        if self.y is not None:
                            # Save argmax of y batch
                            if self.time_axis == 2:
                                img_y = np.argmax(batch_y[0][i, :, frame],
                                                  axis=0)
                                img_channel_axis = 0
                                img_y = batch_y[0][i, :, frame]
                            else:
                                img_channel_axis = -1
                                img_y = batch_y[0][i, frame]
                            img_y = np.argmax(img_y, axis=img_channel_axis)
                            img_y = np.expand_dims(img_y,
                                                   axis=img_channel_axis)
                            img = array_to_img(img_y, self.data_format,
                                               scale=True)
                            fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                                prefix=self.save_prefix,
                                index=j,
                                hash=np.random.randint(1e4),
                                format=self.save_format)
                            img.save(os.path.join(self.save_to_dir, fname))

        def _resize_im(input_im, shape, order):
            dtype = input_im.dtype
            batch_list = []
            for batch_num in range(input_im.shape[0]):
                batch = input_im[batch_num, ...]

                if self.data_format == 'channels_first':

                    batch = np.moveaxis(batch, 0, -1)
                    resized = resize(batch, shape, order=order, preserve_range=True)
                    resized = np.moveaxis(resized, -1, 0)

                    if resized.shape[0] > 1:
                        resized = np.around(resized, decimals=0)

                else:
                    resized = resize(batch, shape, order=order, preserve_range=True)

                    if resized.shape[-1] > 1:
                        resized = np.around(resized, decimals=0)

                batch_list.append(resized)
            return np.stack(batch_list, axis=0).astype(dtype)

        if self.aug_3d and self.rotation_3d > 0:
            scale = tuple([1 / self.z_scale, 1, 1])
            out_shape = tuple([self.output_frames, self.frame_shape[0], self.frame_shape[1]])

            batch_x = _resize_im(batch_x, out_shape, order=1)

            for y in range(len(batch_y)):
                batch_y[y] = _resize_im(batch_y[y], out_shape, order=0)

        return batch_x, batch_y

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


class Semantic3DGenerator(ImageDataGenerator):
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
    def __init__(self, **kwargs):
        super(Semantic3DGenerator, self).__init__(**kwargs)
        # Change the axes for 5D data
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 3
            self.col_axis = 4
            self.time_axis = 2
        if self.data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.time_axis = 1

    def flow(self,
             train_dict,
             batch_size=1,
             frames_per_batch=5,
             frame_shape=None,
             transforms=['outer-distance'],
             transforms_kwargs={},
             aug_3d=False,
             rotation_3d=0,
             z_scale=None,
             shuffle=True,
             min_objects=3,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict (dict): Consists of numpy arrays for ``X`` and ``y``.
            batch_size (int): Size of a batch.
            frames_per_batch (int): Size of z axis in generated batches.
            shuffle (bool): Whether to shuffle the data between epochs.
            seed (int): Random seed for data shuffling.
            min_objects (int): Images with fewer than ``min_objects``
                are ignored.
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if ``save_to_dir`` is set).
            save_format (str): Format to use for saving sample images
                (if ``save_to_dir`` is set).

        Returns:
            Semantic3DIterator: An ``Iterator`` yielding tuples of ``(x, y)``,
            where ``x`` is a numpy array of image data and ``y`` is list of
            numpy arrays of transformed masks of the same shape.
        """
        return Semantic3DIterator(
            train_dict,
            self,
            batch_size=batch_size,
            frames_per_batch=frames_per_batch,
            frame_shape=frame_shape,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            aug_3d=aug_3d,
            rotation_3d=rotation_3d,
            z_scale=z_scale,
            shuffle=shuffle,
            min_objects=min_objects,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        Args:
            x (numpy.array): batch of inputs to be normalized.

        Returns:
            numpy.array: The normalized inputs.
        """
        # TODO: standardize each image, not all frames at once
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) +
                  K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it '
                                'hasn\'t been fit on any training data. '
                                'Fit it first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it '
                                'hasn\'t been fit on any training data. Fit '
                                'it first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`zca_whitening`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
        return x

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        Args:
            x (numpy.array): The data to fit on. Should have rank 5.
            augment (bool): Whether to fit on randomly augmented samples.
            rounds (bool): If augment,
                how many augmentation passes to do over the data.
            seed (int): Random seed for data shuffling.
        Raises:
            ValueError: If input rank is not 5.
            ImportError: If zca_whitening is used and scipy is not available.
        """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))

        if x.shape[self.channel_axis] not in {1, 3, 4}:
            logging.warning(
                'Expected input to be images (as Numpy array) following the '
                'data format convention "{0}" (channels on axis {1}), i.e. '
                'expected either 1, 3, or 4 channels on axis {1}. '
                'However, it was passed an array with shape {2} ({3}) '
                'channels.'.format(
                    self.data_format,
                    self.channel_axis,
                    x.shape,
                    x.shape[self.channel_axis]
                ))

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                          dtype=self.dtype)

            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            axis = (0, self.time_axis, self.row_axis, self.col_axis)
            self.mean = np.mean(x, axis=axis)
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            axis = (0, self.time_axis, self.row_axis, self.col_axis)
            self.std = np.std(x, axis=axis)
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            if scipy is None:
                raise ImportError('Using zca_whitening requires SciPy. '
                                  'Install SciPy.')
            shape = (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
            flat_x = np.reshape(x, shape)
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = scipy.linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)

    def random_transform(self, x, y=None, seed=None, aug_3d=False, rotation_3d=0):
        """Applies a random transformation to an image.

        Args:
            x (numpy.array): 4D tensor or list of 4D tensors.
            y (numpy.array): 4D tensor or list of 4D tensors,
                label mask(s) for ``x``, optional.
            seed (int): Random seed.

        Returns:
            numpy.array: A randomly transformed copy of the input (same shape).
            If ``y`` is passed, it is transformed if necessary and returned.
        """
        self.row_axis -= 1
        self.col_axis -= 1
        self.time_axis -= 1
        self.channel_axis -= 1

        x = x if isinstance(x, list) else [x]
        params = self.get_random_transform(x[0].shape, seed)

        # Don't want to brighten or zoom multiple times
        _brightness_range = self.brightness_range
        _zoom_range = self.zoom_range
        _rotation_range = self.rotation_range
        self.brightness_range = None
        self.zoom_range = (1, 1)
        self.rotation_range = rotation_3d

        # Set params for 3d_augmentation with rotation set to 0
        # Compatible with anisotropic data (with sampling not 1:1:1)
        params_3d = self.get_random_transform(np.moveaxis(x[0], 0, 1).shape, seed)

        self.brightness_range = _brightness_range
        self.zoom_range = _zoom_range
        self.rotation_range = _rotation_range

        for i in range(len(x)):
            x_i = x[i]
            for frame in range(x_i.shape[self.time_axis]):
                if self.data_format == 'channels_first':
                    x_trans = self.apply_transform(x_i[:, frame], params)
                    x_i[:, frame] = np.rollaxis(x_trans, -1, 0)
                else:
                    x_i[frame] = self.apply_transform(x_i[frame], params)

            if aug_3d:
                for frame in range(x_i.shape[self.row_axis]):
                    if self.data_format == 'channels_first':
                        x_trans = self.apply_transform(x_i[:, :, frame], params_3d)
                        x_i[:, :, frame] = np.rollaxis(x_trans, -1, 0)
                    else:
                        x_i[:, frame] = self.apply_transform(x_i[:, frame], params_3d)

                for frame in range(x_i.shape[self.col_axis]):
                    if self.data_format == 'channels_first':
                        x_trans = self.apply_transform(x_i[..., frame], params_3d)
                        x_i[..., frame] = np.rollaxis(x_trans, -1, 0)
                    else:
                        x_i[:, :, frame] = self.apply_transform(x_i[:, :, frame], params_3d)

            x[i] = x_i

        x = x[0] if len(x) == 1 else x

        if y is not None:
            params['brightness'] = None
            params['channel_shift_intensity'] = None

            _interpolation_order = self.interpolation_order
            y = y if isinstance(y, list) else [y]

            for i in range(len(y)):
                y_i = y[i]

                order = 0 if y_i.shape[self.channel_axis] > 1 else _interpolation_order
                self.interpolation_order = order

                for frame in range(y_i.shape[self.time_axis]):
                    if self.data_format == 'channels_first':
                        y_trans = self.apply_transform(y_i[:, frame], params)
                        y_i[:, frame] = np.rollaxis(y_trans, 1, 0)
                    else:
                        y_i[frame] = self.apply_transform(y_i[frame], params)

                # Augment masks in 3D
                if aug_3d:
                    for frame in range(y_i.shape[self.row_axis]):
                        if self.data_format == 'channels_first':
                            y_trans = self.apply_transform(y_i[:, :, frame], params_3d)
                            y_i[:, :, frame] = np.moveaxis(y_trans, -1, 0)
                        else:
                            y_i[:, frame] = self.apply_transform(y_i[:, frame], params_3d)

                    for frame in range(y_i.shape[self.col_axis]):
                        if self.data_format == 'channels_first':
                            y_trans = self.apply_transform(y_i[..., frame], params_3d)
                            y_i[..., frame] = np.moveaxis(y_trans, -1, 0)
                        else:
                            y_i[:, :, frame] = self.apply_transform(y_i[:, :, frame], params_3d)

                y[i] = y_i

                self.interpolation_order = _interpolation_order

            y = y[0] if len(y) == 1 else y

        # Note: Undo workaround
        self.row_axis += 1
        self.col_axis += 1
        self.time_axis += 1
        self.channel_axis += 1

        if y is None:
            return x

        return x, y
