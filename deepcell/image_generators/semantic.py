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
"""Semantic segmentation data generators."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import warnings

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
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
    """Iterator yielding data from Numpy arrays (X and y).

    Args:
        train_dict (dict): Consists of numpy arrays for X and y.
        image_data_generator (ImageDataGenerator): For random transformations
            and normalization.
        batch_size (int): Size of a batch.
        min_objects (int): Images with fewer than 'min_objects' are ignored.
        shuffle (bool): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        data_format (str): One of 'channels_first', 'channels_last'.
        save_to_dir (str): Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix (str): Prefix to use for saving sample
            images (if save_to_dir is set).
        save_format (str): Format to use for saving sample images
            (if save_to_dir is set).
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
                 save_format='png',
                 crop_size=None):
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

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.min_objects = min_objects

        self.y_semantic_list = []  # optional semantic segmentation targets

        # set crop size based on current image if not specified
        if crop_size is not None:
            img_dims = X.shape[1:3] if self.data_format == 'channels_last' else X.shape[2:4]
            print(img_dims, crop_size)

            if img_dims[0] < crop_size[0] or img_dims[1] < crop_size[1]:
                raise ValueError('Crop dimensions must be smaller than image dimensions')

        else:
            crop_size = self.x.shape[1:3] if self.channel_axis == 3 else self.x.shape[2:4]

        self.crop_size = crop_size
        # Create a list of all the semantic targets. We need to be able
        # to have multiple semantic heads
        # Add all the keys that contain y_semantic

        # Add transformed masks
        for transform in transforms:
            transform_kwargs = transforms_kwargs.get(transform, dict())
            y_transform = _transform_masks(y, transform,
                                           data_format=data_format,
                                           **transform_kwargs)
            if y_transform.shape[self.channel_axis] > 1:
                y_transform = np.asarray(y_transform, dtype='int32')
            elif y_transform.shape[self.channel_axis] == 1:
                y_transform = np.asarray(y_transform, dtype=K.floatx())
            self.y_semantic_list.append(y_transform)

        invalid_batches = []

        # Remove images with small numbers of cells
        for b in range(self.x.shape[0]):
            y_batch = np.squeeze(self.y[b], axis=self.channel_axis - 1)
            y_batch = np.expand_dims(y_batch, axis=self.channel_axis - 1)

            self.y[b] = y_batch

            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.x = np.delete(self.x, invalid_batches, axis=0)
        self.y = np.delete(self.y, invalid_batches, axis=0)
        self.y_semantic_list = [np.delete(y, invalid_batches, axis=0)
                                for y in self.y_semantic_list]

        super(SemanticIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # set output shape based on crop shape and image shape
        if self.channel_axis == 3:
            x_shape = tuple([len(index_array)] + list(self.crop_size) + [self.x.shape[3]])
        else:
            x_shape = tuple([len(index_array)] + [self.x.shape[1]] + list(self.crop_size))

        batch_x = np.zeros(x_shape)
        batch_y = []
        for y_sem in self.y_semantic_list:
            # set output shape based on crop shape and transformed label shape
            if self.channel_axis == 3:
                y_shape = tuple([len(index_array)] + list(self.crop_size) + [y_sem.shape[3]])
            else:
                y_shape = tuple([len(index_array)] + [y_sem.shape[1]] + list(self.crop_size))

            batch_y.append(np.zeros(y_shape, dtype=y_sem.dtype))

        for i, j in enumerate(index_array):
            x = self.x[j]

            y_semantic_list = [y_sem[j] for y_sem in self.y_semantic_list]

            # Apply transformation
            x, y_semantic_list = self.image_data_generator.random_transform(
                x, y_semantic_list)

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

            for k, y_sem in enumerate(y_semantic_list):
                batch_y[k][i] = y_sem

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
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1], while with
              width_shift_range=1.0 possible values are floats in the interval
              [-1.0, +1.0).

        shear_range (float): Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range (float): float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range (float): range for random channel shifts.
        fill_mode (str): One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:

                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval (float): Value used for points outside the boundaries
            when fill_mode = "constant".
        horizontal_flip (bool): Randomly flip inputs horizontally.
        vertical_flip (bool): Randomly flip inputs vertically.
        rescale (float): rescaling factor. Defaults to None. If None or 0, no
            rescaling is applied, otherwise we multiply the data by the value
            provided (before applying any other transformation).
        preprocessing_function (function): function that will be implied on
            each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format (str): One of {"channels_first", "channels_last"}.

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

        validation_split (float): Fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format='channels_last',
                 validation_split=0.0,
                 # interpolation_order=1,
                 crop_size=None,
                 dtype='float32'):

        if crop_size is not None:
            if not isinstance(crop_size, (tuple, list)):
                raise ValueError("Crop size must be a list or tuple of row/col dimensions")

        self.crop_size = crop_size

        ImageDataGenerator.__init__(self,
                                    featurewise_center=featurewise_center,
                                    samplewise_center=samplewise_center,
                                    featurewise_std_normalization=featurewise_std_normalization,
                                    samplewise_std_normalization=samplewise_std_normalization,
                                    zca_whitening=zca_whitening,
                                    zca_epsilon=zca_epsilon,
                                    rotation_range=rotation_range,
                                    width_shift_range=width_shift_range,
                                    height_shift_range=height_shift_range,
                                    brightness_range=brightness_range,
                                    shear_range=shear_range,
                                    zoom_range=zoom_range,
                                    channel_shift_range=channel_shift_range,
                                    fill_mode=fill_mode,
                                    cval=cval,
                                    horizontal_flip=horizontal_flip,
                                    vertical_flip=vertical_flip,
                                    rescale=rescale,
                                    preprocessing_function=preprocessing_function,
                                    data_format=data_format,
                                    validation_split=validation_split,
                                    # interpolation_order=interpolation_order,
                                    dtype=dtype)

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
            train_dict (dict): Consists of numpy arrays for X and y.
            batch_size (int): Size of a batch. Defaults to 1.
            shuffle (bool): Whether to shuffle the data between epochs.
                Defaults to True
            seed (int): Random seed for data shuffling.
            min_objects (int): Minumum number of objects allowed per image
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if save_to_dir is set).
            save_format (str): Format to use for saving sample images
                (if save_to_dir is set).

        Returns:
            SemanticIterator: An Iterator yielding tuples of (x, y),
                where x is a numpy array of image data and y is list of
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
            save_format=save_format,
            crop_size=self.crop_size)

    def get_random_transform(self, img_shape, seed=None):
        transform_parameters = ImageDataGenerator.get_random_transform(self, img_shape=img_shape,
                                                                       seed=seed)

        crop_indices = None
        if self.crop_size is not None:
            row_start = np.random.randint(0, img_shape[0] - self.crop_size[0])
            col_start = np.random.randint(0, img_shape[1] - self.crop_size[1])
            crop_indices = ([row_start, row_start + self.crop_size[0]],
                            [col_start, col_start + self.crop_size[1]])

        transform_parameters['crop_indices'] = crop_indices

        return transform_parameters

    def apply_transform(self, x, transform_parameters):

        if transform_parameters['crop_indices'] is not None:
            row_indices, col_indices = transform_parameters['crop_indices']
            x = x[row_indices[0]:row_indices[1], col_indices[0]:col_indices[1]]

        x = ImageDataGenerator.apply_transform(self,
                                               x=x,
                                               transform_parameters=transform_parameters)
        return x

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x (numpy.array): 3D tensor or list of 3D tensors,
                single image.
            y (numpy.array): 3D tensor or list of 3D tensors,
                label mask(s) for x, optional.
            seed (int): Random seed.

        Returns:
            numpy.array: A randomly transformed copy of the input (same shape).
                If y is passed, it is transformed if necessary and returned.
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

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits the data generator to some sample data.
        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.
        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening` are set to True.
        When `rescale` is set to a value, rescaling is applied to
        sample data before computing the internal data stats.
        # Arguments
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
       """
        x = np.asarray(x, dtype=self.dtype)
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
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
        if self.rescale:
            x *= self.rescale

        if augment:
            # adjust output shape to account for cropping in generator
            if self.crop_size is not None:
                if self.channel_axis == 1:
                    x_crop_shape = [x.shape[1]] + list(self.crop_size)
                else:
                    x_crop_shape = list(self.crop_size) + [x.shape[3]]

                ax = np.zeros(
                    tuple([rounds * x.shape[0]] + x_crop_shape),
                    dtype=self.dtype)
            else:
                ax = np.zeros(
                    tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                    dtype=self.dtype)

            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        # TODO: Determine if we want to just call Super__fit() or keep the code below here
        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + 1e-6)

        if self.zca_whitening:
            if scipy is None:
                raise ImportError('Using zca_whitening requires SciPy. '
                                  'Install SciPy.')
            flat_x = np.reshape(
                x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)


class SemanticMovieIterator(Iterator):
    """Iterator yielding data from Numpy arrays (X and y).

    Args:
        train_dict (dict): Dictionary consisting of numpy arrays for X and y.
        movie_data_generator (SemanticMovieGenerator): SemanticMovieGenerator
            to use for random transformations and normalization.
        batch_size (int): Size of a batch.
        frames_per_batch (int): Size of z axis in generated batches.
        shuffle (boolean): Whether to shuffle the data between epochs.
        seed (int): Random seed for data shuffling.
        min_objects (int): Minumum number of objects allowed per image.
        data_format (str): One of 'channels_first', 'channels_last'.
        save_to_dir (str): Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix (str): Prefix to use for saving sample
            images (if save_to_dir is set).
        save_format (str): Format to use for saving sample images
            (if save_to_dir is set).
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

        self.y_semantic_list = []  # optional semantic segmentation targets

        if X.shape[self.time_axis] - frames_per_batch < 0:
            raise ValueError(
                'The number of frames used in each training batch should '
                'be less than the number of frames in the training data!')

        # Create a list of all the semantic targets. We need to be able
        # to have multiple semantic heads
        # Add all the keys that contain y_semantic

        # Add transformed masks
        for transform in transforms:
            transform_kwargs = transforms_kwargs.get(transform, dict())
            y_transform = _transform_masks(y, transform,
                                           data_format=data_format,
                                           **transform_kwargs)
            if y_transform.shape[self.channel_axis] > 1:
                y_transform = np.asarray(y_transform, dtype='int32')
            elif y_transform.shape[self.channel_axis] == 1:
                y_transform = np.asarray(y_transform, dtype=K.floatx())
            self.y_semantic_list.append(y_transform)

        invalid_batches = []

        # Remove images with small numbers of cells
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
        self.y_semantic_list = [np.delete(y, invalid_batches, axis=0)
                                for y in self.y_semantic_list]

        super(SemanticMovieIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            batch_x = np.zeros((len(index_array),
                                self.x.shape[1],
                                self.frames_per_batch,
                                self.x.shape[3],
                                self.x.shape[4]))
        else:
            batch_x = np.zeros(tuple([len(index_array),
                                     self.frames_per_batch] +
                                     list(self.x.shape)[2:]))

        if self.data_format == 'channels_first':
            batch_y_semantic_list = [np.zeros(tuple([len(index_array),
                                                     y_semantic.shape[1],
                                                     self.frames_per_batch,
                                                     y_semantic.shape[3],
                                                     y_semantic.shape[4]]))
                                     for y_semantic in self.y_semantic_list]
        else:
            batch_y_semantic_list = [
                np.zeros(tuple([len(index_array), self.frames_per_batch] +
                               list(y_semantic.shape[2:])))
                for y_semantic in self.y_semantic_list
            ]
        for i, j in enumerate(index_array):
            last_frame = self.x.shape[self.time_axis] - self.frames_per_batch
            time_start = np.random.randint(0, high=last_frame)
            time_end = time_start + self.frames_per_batch

            if self.time_axis == 1:
                x = self.x[j, time_start:time_end, ...]
                y = self.y[j, time_start:time_end, ...]
            elif self.time_axis == 2:
                x = self.x[j, :, time_start:time_end, ...]
                y = self.y[j, :, time_start:time_end, ...]

            if self.time_axis == 1:
                y_semantic_list = [y_semantic[j, time_start:time_end, ...]
                                   for y_semantic in self.y_semantic_list]
            elif self.time_axis == 2:
                y_semantic_list = [y_semantic[j, :, time_start:time_end, ...]
                                   for y_semantic in self.y_semantic_list]

            # Apply transformation
            x, y_list = self.movie_data_generator.random_transform(x, [y] + y_semantic_list)
            y = y_list[0]
            y_semantic_list = y_list[1:]

            x = self.movie_data_generator.standardize(x)

            batch_x[i] = x

            for k, y_sem in enumerate(y_semantic_list):
                batch_y_semantic_list[k][i] = y_sem

            batch_y = batch_y_semantic_list

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
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1], while with
              width_shift_range=1.0 possible values are floats in the interval
              [-1.0, +1.0).

        shear_range (float): Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range (float): float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range (float): range for random channel shifts.
        fill_mode (str): One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:

                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval (float): Value used for points outside the boundaries
            when fill_mode = "constant".
        horizontal_flip (bool): Randomly flip inputs horizontally.
        vertical_flip (bool): Randomly flip inputs vertically.
        rescale (float): rescaling factor. Defaults to None. If None or 0, no
            rescaling is applied, otherwise we multiply the data by the value
            provided (before applying any other transformation).
        preprocessing_function (function): function that will be implied on
            each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format (str): One of {"channels_first", "channels_last"}.

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

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
            train_dict (dict): Consists of numpy arrays for X and y.
            batch_size (int): Size of a batch.
            frames_per_batch (int): Size of z axis in generated batches.
            shuffle (bool): Whether to shuffle the data between epochs.
            seed (int): Random seed for data shuffling.
            min_objects (int): Images with fewer than 'min_objects' are ignored.
            save_to_dir (str): Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix (str): Prefix to use for saving sample
                images (if save_to_dir is set).
            save_format (str): Format to use for saving sample images
                (if save_to_dir is set).

        Returns:
            SemanticMovieIterator: An Iterator yielding tuples of (x, y)
                where x is a numpy array of image data and y is a list of
                numpy arrays of transformed masks.
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
                If y is passed, it is transformed if necessary and returned.
        """
        self.row_axis -= 1
        self.col_axis -= 1
        self.time_axis -= 1
        self.channel_axis -= 1

        if isinstance(x, list):
            params = self.get_random_transform(x[0].shape, seed)
        else:
            params = self.get_random_transform(x.shape, seed)

        if isinstance(x, list):
            for i in range(len(x)):
                x_i = x[i]
                for frame in range(x_i.shape[self.time_axis]):
                    if self.data_format == 'channels_first':
                        x_trans = self.apply_transform(x_i[:, frame], params)
                        x_i[:, frame] = np.rollaxis(x_trans, -1, 0)
                    else:
                        x_i[frame] = self.apply_transform(x_i[frame], params)
                x[i] = x_i
        else:
            for frame in range(x.shape[self.time_axis]):
                if self.data_format == 'channels_first':
                    x_trans = self.apply_transform(x[:, frame], params)
                    x[:, frame] = np.rollaxis(x_trans, -1, 0)
                else:
                    temp = self.apply_transform(x[frame], params)
                    x[frame] = self.apply_transform(x[frame], params)

        if y is not None:
            params['brightness'] = None
            params['channel_shift_intensity'] = None

            _interpolation_order = self.interpolation_order

            if isinstance(y, list):
                for i in range(len(y)):
                    y_i = y[i]

                    if y_i.shape[self.channel_axis] > 1:
                        self.interpolation_order = 0
                    else:
                        self.interpolation_order = _interpolation_order

                    for frame in range(y_i.shape[self.time_axis]):
                        if self.data_format == 'channels_first':
                            y_trans = self.apply_transform(y_i[:, frame],
                                                           params)
                            y_i[:, frame] = np.rollaxis(y_trans, 1, 0)
                        else:
                            y_i[frame] = self.apply_transform(y_i[frame],
                                                              params)

                    y[i] = y_i

                    self.interpolation_order = _interpolation_order

            else:
                if y.shape[self.channel_axis] > 1:
                    self.interpolation_order = 0
                else:
                    self.interpolation_order = _interpolation_order

                for frame in range(y.shape[self.time_axis]):
                    if self.data_format == 'channels_first':
                        y_trans = self.apply_transform(y[:, frame], params)
                        y[:, frame] = np.rollaxis(y_trans, 1, 0)
                    else:
                        y[frame] = self.apply_transform(y[frame], params)

                self.interpolation_order = _interpolation_order

        # Note: Undo workaround
        self.row_axis += 1
        self.col_axis += 1
        self.time_axis += 1
        self.channel_axis += 1

        if y is None:
            return x

        return x, y
