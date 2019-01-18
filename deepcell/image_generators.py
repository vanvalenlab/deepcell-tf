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
"""Image generators for training convolutional neural networks."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from fnmatch import fnmatch

import cv2
import keras_preprocessing
import numpy as np
import skimage.measure

from skimage.measure import label
from skimage.transform import resize
from skimage.io import imread

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import random_channel_shift
from tensorflow.python.keras.preprocessing.image import apply_transform
from tensorflow.python.keras.preprocessing.image import flip_axis
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


try:
    from tensorflow.python.keras.utils import conv_utils
except ImportError:
    from tensorflow.python.keras._impl.keras.utils import conv_utils

from keras_retinanet.preprocessing.generator import Generator as _RetinaNetGenerator
from keras_maskrcnn.preprocessing.generator import Generator as _MaskRCNNGenerator

from deepcell.utils.data_utils import sample_label_matrix
from deepcell.utils.data_utils import sample_label_movie
from deepcell.utils.transform_utils import transform_matrix_offset_center
from deepcell.utils.transform_utils import deepcell_transform
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.utils.transform_utils import distance_transform_3d
from deepcell.utils.retinanet_anchor_utils import anchor_targets_bbox


def _transform_masks(y, transform, data_format=None, **kwargs):
    """Based on the transform key, apply a transform function to the masks.

    More detailed description. Caution for unknown transorm keys.

    Args:
        y: `labels` of ndim 4 or 5
        transform: one of {`deepcell`, `disc`, `watershed`, `centroid`, `None`}

    Returns:
        y_transform: the output of the given transform function on y

    Raises:
        IOError: An error occurred
    """
    valid_transforms = {'deepcell', 'disc', 'watershed', 'centroid', 'fgbg'}

    if data_format is None:
        data_format = K.image_data_format()

    if y.ndim not in {4, 5}:
        raise ValueError('`labels` data must be of ndim 4 or 5.  Got', y.ndim)

    channel_axis = 1 if data_format == 'channels_first' else -1

    if y.shape[channel_axis] != 1:
        raise ValueError('Expected channel axis to be 1 dimension. Got',
                         y.shape[1 if data_format == 'channels_first' else -1])

    if isinstance(transform, str):
        transform = transform.lower()
        if transform not in valid_transforms:
            raise ValueError('`{}` is not a valid transform'.format(transform))

    if transform == 'deepcell':
        dilation_radius = kwargs.pop('dilation_radius', None)
        y_transform = deepcell_transform(y, dilation_radius, data_format=data_format)

    elif transform == 'watershed':
        distance_bins = kwargs.pop('distance_bins', 4)
        erosion = kwargs.pop('erosion_width', 0)

        if data_format == 'channels_first':
            y_transform = np.zeros((y.shape[0], *y.shape[2:]))
        else:
            y_transform = np.zeros(y.shape[0:-1])

        if y.ndim == 5:
            _distance_transform = distance_transform_3d
        else:
            _distance_transform = distance_transform_2d

        for batch in range(y_transform.shape[0]):
            if data_format == 'channels_first':
                mask = y[batch, 0, ...]
            else:
                mask = y[batch, ..., 0]

            y_transform[batch] = _distance_transform(
                mask, distance_bins, erosion)

        # convert to one hot notation
        y_transform = np.expand_dims(y_transform, axis=-1)
        y_transform = to_categorical(y_transform, num_classes=distance_bins)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'disc':
        y_transform = to_categorical(y.squeeze(channel_axis))
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'fgbg':
        y_transform = np.where(y > 1, 1, y)
        # convert to one hot notation
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, 1, y.ndim)
        y_transform = to_categorical(y_transform)
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform is None:
        y_transform = to_categorical(y.squeeze(channel_axis))
        if data_format == 'channels_first':
            y_transform = np.rollaxis(y_transform, y.ndim - 1, 1)

    elif transform == 'centroid':
        raise NotImplementedError('`centroid` transform has not been finished')

    return y_transform


class ImageSampleArrayIterator(Iterator):
    """Iterator yielding data from a sampled Numpy array.
    Sampling will generate a `window_size` image classifying the center pixel,

    Arguments:
        train_dict: dictionary consisting of numpy arrays for `X` and `y`.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        window_size: size of sampling window around each pixel
        balance_classes: balance class representation when sampling
        max_class_samples: maximum number of samples per class.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=32,
                 shuffle=False,
                 window_size=(30, 30),
                 transform=None,
                 transform_kwargs={},
                 balance_classes=False,
                 max_class_samples=None,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(X, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageSampleArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        window_size = conv_utils.normalize_tuple(window_size, 2, 'window_size')

        y = _transform_masks(y, transform, data_format=data_format, **transform_kwargs)

        pixels_x, pixels_y, batch, y = sample_label_matrix(
            y=y,
            padding='valid',
            window_size=window_size,
            max_training_examples=None,
            data_format=data_format)

        self.y = y
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.batch = batch
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.win_x = window_size[0]
        self.win_y = window_size[1]
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.class_balance(max_class_samples, balance_classes, seed=seed)

        self.y = to_categorical(self.y).astype('int32')
        super(ImageSampleArrayIterator, self).__init__(
            len(self.y), batch_size, shuffle, seed)

    def _sample_image(self, b, px, py):
        wx = self.win_x
        wy = self.win_y

        if self.channel_axis == 1:
            sampled = self.x[b, :, px - wx:px + wx + 1, py - wy:py + wy + 1]
        else:
            sampled = self.x[b, px - wx:px + wx + 1, py - wy:py + wy + 1, :]

        return sampled

    def class_balance(self, max_class_samples=None, downsample=False, seed=None):
        """Balance classes based on the number of samples of each class.

        Args:
            max_class_samples: if not None, a maximum count for each class
            downsample: if True, all sample sizes will be the rarest count
            seed: random state initalization

        Returns:
            Does not return anything but shuffles and resizes the sample size
        """
        balanced_indices = []

        unique_b = np.unique(self.batch)

        if max_class_samples is not None:
            max_class_samples = int(max_class_samples // len(unique_b))

        for b in unique_b:
            batch_y = self.y[self.batch == b]
            unique, counts = np.unique(batch_y, return_counts=True)
            min_index = np.argmin(counts)
            n_samples = counts[min_index]

            if max_class_samples is not None and max_class_samples < n_samples:
                n_samples = max_class_samples

            for class_label in unique:
                non_rand_ind = ((self.batch == b) & (self.y == class_label)).nonzero()[0]

                if downsample:
                    size = n_samples
                elif max_class_samples:
                    size = min(max_class_samples, len(non_rand_ind))
                else:
                    size = len(non_rand_ind)

                index = np.random.choice(non_rand_ind, size=size, replace=False)
                balanced_indices.extend(index)

        np.random.seed(seed=seed)
        np.random.shuffle(balanced_indices)

        self.batch = self.batch[balanced_indices]
        self.pixels_x = self.pixels_x[balanced_indices]
        self.pixels_y = self.pixels_y[balanced_indices]
        self.y = self.y[balanced_indices]

    def _get_batches_of_transformed_samples(self, index_array):
        if self.channel_axis == 1:
            batch_x = np.zeros((len(index_array),
                                self.x.shape[self.channel_axis],
                                2 * self.win_x + 1,
                                2 * self.win_y + 1))
        else:
            batch_x = np.zeros((len(index_array),
                                2 * self.win_x + 1,
                                2 * self.win_y + 1,
                                self.x.shape[self.channel_axis]))

        for i, j in enumerate(index_array):
            b, px, py = self.batch[j], self.pixels_x[j], self.pixels_y[j]
            x = self._sample_image(b, px, py)
            x = self.image_data_generator.random_transform(x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
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


class SampleDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Arguments:
        featurewise_center: boolean, set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center: boolean, set each sample mean to 0.
        featurewise_std_normalization: boolean, divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization: boolean, divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: boolean, apply ZCA whitening.
        rotation_range: int, degree range for random rotations.
        width_shift_range: float, 1-D array-like or int
            float: fraction of total width, if < 1, or pixels if >= 1.
            1-D array-like: random elements from the array.
            int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            With `width_shift_range=2` possible values are ints [-1, 0, +1],
            same as with `width_shift_range=[-1, 0, +1]`,
            while with `width_shift_range=1.0` possible values are floats in
            the interval [-1.0, +1.0).
        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: float or int, value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: boolean, randomly flip inputs horizontally.
        vertical_flip: boolean, randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: One of {"channels_first", "channels_last"}.
            "channels_last" mode means that the images should have shape
                `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
                `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: float, fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def flow(self,
             train_dict,
             batch_size=32,
             shuffle=True,
             transform=None,
             transform_kwargs={},
             window_size=(30, 30),
             balance_classes=False,
             max_class_samples=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Arguments:
            train_dict: dictionary consisting of numpy arrays for `X` and `y`.
            image_data_generator: Instance of `ImageDataGenerator`
                to use for random transformations and normalization.
            batch_size: Integer, size of a batch.
            shuffle: Boolean, whether to shuffle the data between epochs.
            window_size: size of sampling window around each pixel
            balance_classes: balance class representation when sampling
            max_class_samples: maximum number of samples per class.
            seed: Random seed for data shuffling.
            data_format: String, one of `channels_first`, `channels_last`.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if `save_to_dir` is set).
            save_format: Format to use for saving sample images
                (if `save_to_dir` is set).
        """
        return ImageSampleArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=transform,
            transform_kwargs=transform_kwargs,
            window_size=window_size,
            balance_classes=balance_classes,
            max_class_samples=max_class_samples,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class ImageFullyConvIterator(Iterator):
    """Iterator yielding data from Numpy arrayss (`X and `y`).

    Arguments:
        train_dict: dictionary consisting of numpy arrays for `X` and `y`.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 skip=None,
                 shuffle=False,
                 transform=None,
                 transform_kwargs={},
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(X, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageFullyConvIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.y = _transform_masks(y, transform, data_format=data_format, **transform_kwargs)
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.skip = skip
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImageFullyConvIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:]))

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img_x = np.expand_dims(batch_x[i, :, :, 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.y is not None:
                    # Save argmax of y batch
                    img_y = np.argmax(batch_y[i], axis=self.channel_axis - 1)
                    img_y = np.expand_dims(img_y, axis=self.channel_axis - 1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x

        if self.skip is not None:
            batch_y = [batch_y] * (self.skip + 1)
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


class ImageFullyConvDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Arguments:
        featurewise_center: boolean, set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center: boolean, set each sample mean to 0.
        featurewise_std_normalization: boolean, divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization: boolean, divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: boolean, apply ZCA whitening.
        rotation_range: int, degree range for random rotations.
        width_shift_range: float, 1-D array-like or int
            float: fraction of total width, if < 1, or pixels if >= 1.
            1-D array-like: random elements from the array.
            int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            With `width_shift_range=2` possible values are ints [-1, 0, +1],
            same as with `width_shift_range=[-1, 0, +1]`,
            while with `width_shift_range=1.0` possible values are floats in
            the interval [-1.0, +1.0).
        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: float or int, value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: boolean, randomly flip inputs horizontally.
        vertical_flip: boolean, randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: One of {"channels_first", "channels_last"}.
            "channels_last" mode means that the images should have shape
                `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
                `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: float, fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def flow(self,
             train_dict,
             batch_size=1,
             skip=None,
             transform=None,
             transform_kwargs={},
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Arguments:
            train_dict: dictionary of X and y tensors. Both should be rank 4.
            batch_size: int (default: 1).
            shuffle: boolean (default: True).
            seed: int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: str (default: `''`). Prefix to use for filenames of
                saved pictures (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg". Default: "png".
                (only relevant if `save_to_dir` is set)

        Returns:
            An Iterator yielding tuples of `(x, y)` where `x` is a numpy array
            of image data and `y` is a numpy array of labels of the same shape.
        """
        return ImageFullyConvIterator(
            train_dict,
            self,
            batch_size=batch_size,
            transform=transform,
            transform_kwargs=transform_kwargs,
            skip=skip,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        Args:
            x: batch of inputs to be normalized.

        Returns:
            The normalized inputs.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + K.epsilon())

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + K.epsilon())
            else:
                logging.warning('This ImageDataGenerator specifies '
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x, labels=None, seed=None):
        """Randomly augment a single image tensor and its labels.

        Args:
            x: 4D tensor, single image.
            labels: 4D tensor, single image mask.
            seed: random seed.

        Returns:
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.deg2rad(
                np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(
                transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([
                [1, -np.sin(shear), 0],
                [0, np.cos(shear), 0],
                [0, 0, 1]
            ])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(
                transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([
                [zx, 0, 0],
                [0, zy, 0],
                [0, 0, 1]
            ])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
                transform_matrix, zoom_matrix)

        if labels is not None:
            y = labels  # np.expand_dims(labels, axis=0)

            if transform_matrix is not None:
                h, w = y.shape[img_row_axis], y.shape[img_col_axis]
                transform_matrix_y = transform_matrix_offset_center(transform_matrix, h, w)
                y = apply_transform(y, transform_matrix_y, img_channel_axis,
                                    fill_mode='constant', cval=0)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix_x = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix_x, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                if labels is not None:
                    y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                if labels is not None:
                    y = flip_axis(y, img_row_axis)

        if labels is not None:
            return x, y.astype('int')
        return x


class MovieDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Arguments:
        featurewise_center: boolean, set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center: boolean, set each sample mean to 0.
        featurewise_std_normalization: boolean, divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization: boolean, divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: boolean, apply ZCA whitening.
        rotation_range: int, degree range for random rotations.
        width_shift_range: float, 1-D array-like or int
            float: fraction of total width, if < 1, or pixels if >= 1.
            1-D array-like: random elements from the array.
            int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            With `width_shift_range=2` possible values are ints [-1, 0, +1],
            same as with `width_shift_range=[-1, 0, +1]`,
            while with `width_shift_range=1.0` possible values are floats in
            the interval [-1.0, +1.0).
        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: float or int, value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: boolean, randomly flip inputs horizontally.
        vertical_flip: boolean, randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: One of {"channels_first", "channels_last"}.
            "channels_last" mode means that the images should have shape
                `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
                `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: float, fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def __init__(self, **kwargs):
        super(MovieDataGenerator, self).__init__(**kwargs)
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
             frames_per_batch=10,
             skip=None,
             transform=None,
             transform_kwargs={},
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Arguments:
            train_dict: dictionary of X and y tensors. Both should be rank 5.
            frames_per_batch: int (default: 10).
                size of z axis in generated batches
            batch_size: int (default: 1).
            shuffle: boolean (default: True).
            seed: int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: str (default: `''`). Prefix to use for filenames of
                saved pictures (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg". Default: "png".
                (only relevant if `save_to_dir` is set)

        Returns:
            An Iterator yielding tuples of `(x, y)` where `x` is a numpy array
            of image data and `y` is a numpy array of labels of the same shape.
        """
        return MovieArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            frames_per_batch=frames_per_batch,
            skip=skip,
            transform=transform,
            transform_kwargs=transform_kwargs,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        Args:
            x: batch of inputs to be normalized.

        Returns:
            The normalized inputs.
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
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + K.epsilon())

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
                                'hasn\'t been fit on any training data. '
                                'Fit it first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x, labels=None, seed=None):
        """Randomly augment a single image tensor and its labels.

        Args:
            x: 5D tensor, image stack.
            labels: 5D tensor, image mask stack.
            seed: random seed.

        Returns:
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_time_axis = self.time_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.deg2rad(
                np.random.uniform(-self.rotation_range, self.rotation_range))
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            if self.width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(
                transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([
                [1, -np.sin(shear), 0],
                [0, np.cos(shear), 0],
                [0, 0, 1]
            ])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(
                transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([
                [zx, 0, 0],
                [0, zy, 0],
                [0, 0, 1]
            ])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
                transform_matrix, zoom_matrix)

        if labels is not None:
            y = labels

            if transform_matrix is not None:
                y_new = []
                h, w = y.shape[img_row_axis], y.shape[img_col_axis]
                transform_matrix_y = transform_matrix_offset_center(transform_matrix, h, w)
                for frame in range(y.shape[img_time_axis]):
                    if self.time_axis == 2:
                        y_frame = y[:, frame]
                        trans_channel_axis = img_channel_axis
                    else:
                        y_frame = y[frame]
                        trans_channel_axis = img_channel_axis - 1
                    y_trans = apply_transform(y_frame, transform_matrix_y, trans_channel_axis,
                                              fill_mode='constant', cval=0)
                    y_new.append(np.rint(y_trans))
                y = np.stack(y_new, axis=img_time_axis)

        if transform_matrix is not None:
            x_new = []
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix_x = transform_matrix_offset_center(transform_matrix, h, w)
            for frame in range(x.shape[img_time_axis]):
                if self.time_axis == 2:
                    x_frame = x[:, frame]
                    trans_channel_axis = img_channel_axis
                else:
                    x_frame = x[frame]
                    trans_channel_axis = img_channel_axis - 1
                x_trans = apply_transform(x_frame, transform_matrix_x, trans_channel_axis,
                                          fill_mode=self.fill_mode, cval=self.cval)
                x_new.append(x_trans)
            x = np.stack(x_new, axis=img_time_axis)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                if labels is not None:
                    y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                if labels is not None:
                    y = flip_axis(y, img_row_axis)

        if labels is not None:
            return x, y

        return x

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits internal statistics to some sample data.

        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        Args:
            x: Numpy array, the data to fit on. Should have rank 5.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.

        Raises:
            ValueError: If input rank is not 5.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 5:
            raise ValueError('Input to `.fit()` should have rank 5. '
                             'Got array with shape: ' + str(x.shape))

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.time_axis, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.time_axis, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())


class MovieArrayIterator(Iterator):
    """Iterator yielding data from two 5D Numpy arrays (`X and `y`).

    Arguments:
        train_dict: dictionary consisting of numpy arrays for `X` and `y`.
        movie_data_generator: Instance of `MovieDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        frames_per_batch: size of z axis in generated batches
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self,
                 train_dict,
                 movie_data_generator,
                 batch_size=32,
                 frames_per_batch=10,
                 skip=None,
                 transform=None,
                 transform_kwargs={},
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError('`X` (movie data) and `y` (labels) '
                             'should have the same size. Found '
                             'Found x.shape = {}, y.shape = {}'.format(
                                 X.shape, y.shape))

        if data_format is None:
            data_format = K.image_data_format()

        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.x = np.asarray(X, dtype=K.floatx())
        self.y = _transform_masks(y, transform, data_format=data_format, **transform_kwargs)

        if self.x.ndim != 5:
            raise ValueError('Input data in `MovieArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)

        if self.x.shape[self.time_axis] - frames_per_batch < 0:
            raise ValueError(
                'The number of frames used in each training batch should '
                'be less than the number of frames in the training data!')

        self.frames_per_batch = frames_per_batch
        self.skip = skip
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(MovieArrayIterator, self).__init__(
            len(self.y), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            batch_x = np.zeros((len(index_array),
                                self.x.shape[1],
                                self.frames_per_batch,
                                self.x.shape[3],
                                self.x.shape[4]))
            if self.y is not None:
                batch_y = np.zeros((len(index_array),
                                    self.y.shape[1],
                                    self.frames_per_batch,
                                    self.y.shape[3],
                                    self.y.shape[4]))

        else:
            batch_x = np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                     list(self.x.shape)[2:]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array), self.frames_per_batch] +
                                         list(self.y.shape)[2:]))

        for i, j in enumerate(index_array):
            if self.y is not None:
                y = self.y[j]

            # Sample along the time axis
            last_frame = self.x.shape[self.time_axis] - self.frames_per_batch
            time_start = np.random.randint(0, high=last_frame)
            time_end = time_start + self.frames_per_batch
            if self.time_axis == 1:
                x = self.x[j, time_start:time_end, :, :, :]
                if self.y is not None:
                    y = self.y[j, time_start:time_end, :, :, :]

            elif self.time_axis == 2:
                x = self.x[j, :, time_start:time_end, :, :]
                if self.y is not None:
                    y = self.y[j, :, time_start:time_end, :, :]

            if self.y is not None:
                x, y = self.movie_data_generator.random_transform(
                    x.astype(K.floatx()), labels=y)
                x = self.movie_data_generator.standardize(x)
                batch_y[i] = y
            else:
                x = self.movie_data_generator.random_transform(x.astype(K.floatx()))

            batch_x[i] = x

        if self.save_to_dir:
            time_axis = 2 if self.data_format == 'channels_first' else 1
            for i, j in enumerate(index_array):
                for frame in range(batch_x.shape[time_axis]):
                    if time_axis == 2:
                        img = array_to_img(batch_x[i, :, frame], self.data_format, scale=True)
                    else:
                        img = array_to_img(batch_x[i, frame], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

                    if self.y is not None:
                        # Save argmax of y batch
                        if self.time_axis == 2:
                            img_y = np.argmax(batch_y[i, :, frame], axis=0)
                            img_channel_axis = 0
                            img_y = batch_y[i, :, frame]
                        else:
                            img_channel_axis = -1
                            img_y = batch_y[i, frame]
                        img_y = np.argmax(img_y, axis=img_channel_axis)
                        img_y = np.expand_dims(img_y, axis=img_channel_axis)
                        img = array_to_img(img_y, self.data_format, scale=True)
                        fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                        img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x

        if self.skip is not None:
            batch_y = [batch_y] * (self.skip + 1)

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


class SampleMovieArrayIterator(Iterator):
    """Iterator yielding data from two 5D Numpy arrays (`X and `y`).
    Sampling will generate a `window_size` voxel classifying the center pixel,

    Arguments:
        train_dict: dictionary consisting of numpy arrays for `X` and `y`.
        movie_data_generator: Instance of `MovieDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        window_size: size of sampling window around each pixel
        balance_classes: balance class representation when sampling
        max_class_samples: maximum number of samples per class.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self,
                 train_dict,
                 movie_data_generator,
                 batch_size=32,
                 shuffle=False,
                 transform=None,
                 transform_kwargs={},
                 balance_classes=False,
                 max_class_samples=None,
                 window_size=(30, 30, 5),
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError('`X` (movie data) and `y` (labels) '
                             'should have the same size. Found '
                             'Found x.shape = {}, y.shape = {}'.format(
                                 X.shape, y.shape))

        if data_format is None:
            data_format = K.image_data_format()

        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.x = np.asarray(X, dtype=K.floatx())
        y = _transform_masks(y, transform,
                             data_format=data_format,
                             **transform_kwargs)

        if self.x.ndim != 5:
            raise ValueError('Input data in `SampleMovieArrayIterator` '
                             'should have rank 5. You passed an array '
                             'with shape', self.x.shape)

        window_size = conv_utils.normalize_tuple(window_size, 3, 'window_size')

        pixels_z, pixels_x, pixels_y, batch, y = sample_label_movie(
            y=y,
            padding='valid',
            window_size=window_size,
            max_training_examples=None,
            data_format=data_format)

        self.y = y
        self.win_x = window_size[0]
        self.win_y = window_size[1]
        self.win_z = window_size[2]
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.pixels_z = pixels_z
        self.batch = batch
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.class_balance(max_class_samples, balance_classes, seed=seed)

        self.y = to_categorical(self.y).astype('int32')
        super(SampleMovieArrayIterator, self).__init__(
            len(self.y), batch_size, shuffle, seed)

    def _sample_image(self, b, pz, px, py):
        wx = self.win_x
        wy = self.win_y
        wz = self.win_z

        if self.channel_axis == 1:
            sampled = self.x[b, :, pz - wz:pz + wz + 1, px - wx:px + wx + 1, py - wy:py + wy + 1]
        else:
            sampled = self.x[b, pz - wz:pz + wz + 1, px - wx:px + wx + 1, py - wy:py + wy + 1, :]

        return sampled

    def class_balance(self, max_class_samples=None, downsample=False, seed=None):
        """Balance classes based on the number of samples of each class

        Args:
            max_class_samples: if not None, a maximum count for each class
            downsample: if True, all sample sizes will be the rarest count
            seed: random state initalization
        """
        balanced_indices = []

        unique_b = np.unique(self.batch)

        if max_class_samples is not None:
            max_class_samples = int(max_class_samples // len(unique_b))

        for b in unique_b:
            batch_y = self.y[self.batch == b]
            unique, counts = np.unique(batch_y, return_counts=True)
            min_index = np.argmin(counts)
            n_samples = counts[min_index]

            if max_class_samples is not None and max_class_samples < n_samples:
                n_samples = max_class_samples

            for class_label in unique:
                non_rand_ind = ((self.batch == b) & (self.y == class_label)).nonzero()[0]

                if downsample:
                    size = n_samples
                elif max_class_samples:
                    size = min(max_class_samples, len(non_rand_ind))
                else:
                    size = len(non_rand_ind)

                index = np.random.choice(non_rand_ind, size=size, replace=False)
                balanced_indices.extend(index)

        np.random.seed(seed=seed)
        np.random.shuffle(balanced_indices)

        # Save the upsampled results
        self.batch = self.batch[balanced_indices]
        self.pixels_z = self.pixels_z[balanced_indices]
        self.pixels_x = self.pixels_x[balanced_indices]
        self.pixels_y = self.pixels_y[balanced_indices]
        self.y = self.y[balanced_indices]

    def _get_batches_of_transformed_samples(self, index_array):
        if self.channel_axis == 1:
            batch_x = np.zeros((len(index_array),
                                self.x.shape[self.channel_axis],
                                2 * self.win_z + 1,
                                2 * self.win_x + 1,
                                2 * self.win_y + 1))
        else:
            batch_x = np.zeros((len(index_array),
                                2 * self.win_z + 1,
                                2 * self.win_x + 1,
                                2 * self.win_y + 1,
                                self.x.shape[self.channel_axis]))

        for i, j in enumerate(index_array):
            b, pz, px, py = self.batch[j], self.pixels_z[j], self.pixels_x[j], self.pixels_y[j]
            x = self._sample_image(b, pz, px, py)
            x = self.movie_data_generator.random_transform(x.astype(K.floatx()))
            x = self.movie_data_generator.standardize(x)

            batch_x[i] = x

        if self.save_to_dir:
            time_axis = 2 if self.data_format == 'channels_first' else 1
            for i, j in enumerate(index_array):
                for frame in range(batch_x.shape[time_axis]):
                    if time_axis == 2:
                        img = batch_x[i, :, frame]
                    else:
                        img = batch_x[i, frame]
                    img = array_to_img(img, self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
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


class SampleMovieDataGenerator(MovieDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Arguments:
        featurewise_center: boolean, set input mean to 0 over the dataset,
            feature-wise.
        samplewise_center: boolean, set each sample mean to 0.
        featurewise_std_normalization: boolean, divide inputs by std
            of the dataset, feature-wise.
        samplewise_std_normalization: boolean, divide each input by its std.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        zca_whitening: boolean, apply ZCA whitening.
        rotation_range: int, degree range for random rotations.
        width_shift_range: float, 1-D array-like or int
            float: fraction of total width, if < 1, or pixels if >= 1.
            1-D array-like: random elements from the array.
            int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            With `width_shift_range=2` possible values are ints [-1, 0, +1],
            same as with `width_shift_range=[-1, 0, +1]`,
            while with `width_shift_range=1.0` possible values are floats in
            the interval [-1.0, +1.0).
        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: float or int, value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: boolean, randomly flip inputs horizontally.
        vertical_flip: boolean, randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None. If None or 0, no rescaling
            is applied, otherwise we multiply the data by the value provided
            (before applying any other transformation).
        preprocessing_function: function that will be implied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: One of {"channels_first", "channels_last"}.
            "channels_last" mode means that the images should have shape
                `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
                `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: float, fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def flow(self,
             train_dict,
             batch_size=32,
             shuffle=True,
             transform=None,
             transform_kwargs={},
             window_size=(30, 30, 5),
             balance_classes=False,
             max_class_samples=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Arguments:
            train_dict: dictionary of X and y tensors. Both should be rank 5.
            window_size: tuple (default: (30, 30 5)).
                The size of the sampled voxels to generate.
            batch_size: int (default: 1).
            shuffle: boolean (default: True).
            seed: int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: str (default: `''`). Prefix to use for filenames of
                saved pictures (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg". Default: "png".
                (only relevant if `save_to_dir` is set)

        Returns:
            An Iterator yielding tuples of `(x, y)` where `x` is a numpy array
            of image data and `y` is a numpy array of labels of the same shape.
        """
        return SampleMovieArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=transform,
            transform_kwargs=transform_kwargs,
            window_size=window_size,
            balance_classes=balance_classes,
            max_class_samples=max_class_samples,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


"""
Custom siamese generators
"""


class SiameseDataGenerator(keras_preprocessing.image.ImageDataGenerator):
    def flow(self,
             train_dict,
             crop_dim=32,
             min_track_length=5,
             neighborhood_scale_size=64,
             neighborhood_true_size=100,
             features=None,
             sync_transform=True,
             batch_size=32,
             shuffle=True,
             seed=None,
             data_format=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return SiameseIterator(
            train_dict,
            self,
            crop_dim=crop_dim,
            min_track_length=min_track_length,
            neighborhood_scale_size=neighborhood_scale_size,
            neighborhood_true_size=neighborhood_true_size,
            features=features,
            sync_transform=sync_transform,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

class SiameseIterator(keras_preprocessing.image.Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 crop_dim=32,
                 min_track_length=5,
                 batch_size=32,
                 neighborhood_scale_size=64,
                 neighborhood_true_size=100,
                 features=None,
                 sync_transform=True,
                 shuffle=False,
                 seed=None,
                 squeeze=False,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()

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

        if features is None:
            raise ValueError("SiameseIterator: No features specified.")

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
        self.squeeze = squeeze
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        if 'daughters' in train_dict:
            self.daughters = train_dict['daughters']
        else:
            self.daughters = None

        self._remove_bad_images()
        self._create_track_ids()
        self._create_features()

        super(SiameseIterator, self).__init__(
            len(self.track_ids), batch_size, shuffle, seed)

    def _remove_bad_images(self):
        """
        This function goes through all of the batches of images and removes the
        images that only have one cell.
        """
        good_batches = []
        number_of_batches = self.x.shape[0]
        for batch in range(number_of_batches):
            y = self.y[batch]
            unique_ids = np.unique(y.flatten())
            if len(unique_ids) > 2: # There should be at least 3 id's - 2 cells and 1 background
                good_batches.append(batch)

        X_new_shape = (len(good_batches), *self.x.shape[1:])
        y_new_shape = (len(good_batches), *self.y.shape[1:])

        X_new = np.zeros(X_new_shape, dtype = K.floatx())
        y_new = np.zeros(y_new_shape, dtype = np.int32)

        counter = 0
        for k, batch in enumerate(good_batches):
            X_new[k] = self.x[batch]
            y_new[k] = self.y[batch]

        self.x = X_new
        self.y = y_new
        self.daughters = [self.daughters[i] for i in good_batches]

    def _create_track_ids(self):
        """
        This function builds the track id's. It returns a dictionary that
        contains the batch number and label number of each each track.
        Creates unique cell IDs, as cell labels are NOT unique across batches.

        """
        track_counter = 0
        track_ids = {}
        for batch in range(self.y.shape[0]):
            y_batch = self.y[batch]
            daughters_batch = self.daughters[batch]
            num_cells = np.amax(y_batch)
            for cell in range(1, num_cells + 1):
                # count number of pixels cell occupies in each frame
                y_true = np.sum(y_batch == cell, axis=(self.row_axis - 1, self.col_axis - 1))
                # get indices of frames where cell is present
                y_index = np.where(y_true > 0)[0]
                if y_index.size > 3: # if cell is present at all
                    if self.daughters is not None:
                        # Only include daughters if there are enough frames in their tracks
                        if cell not in daughters_batch:
                            print("something weird...")
                            print("y.shape", self.y.shape)
                            print("unique values in batch:", np.unique(y_batch))
                            print("unique values in y.batch:", np.unique(self.y[batch]))
                            print("loaded lineage cell ids:", daughters_batch.keys())
                            print("batch:", batch)

                        daughter_ids = daughters_batch.get(cell, [])

                        if len(daughter_ids) > 0:
                            daughter_track_lengths = []
                            for did in daughter_ids:
                                # Screen daughter tracks to make sure they are long enough
                                # Length currently set to 0
                                d_true = np.sum(y_batch == did, axis=(self.row_axis - 1, self.col_axis - 1))
                                d_track_length = len(np.where(d_true>0)[0])
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

        # Add a field to the track_ids dict that locates all of the different cells
        # in each frame
        for track in track_ids.keys():
            track_ids[track]['different'] = {}
            batch = track_ids[track]['batch']
            label = track_ids[track]['label']
            for frame in track_ids[track]['frames']:
                y_unique = np.unique(self.y[batch][frame])
                y_unique = np.delete(y_unique, np.where(y_unique == 0))
                y_unique = np.delete(y_unique, np.where(y_unique == label))
                track_ids[track]['different'][frame] = y_unique

        # We will need to look up the track_ids of cells if we know their batch and label. We will
        # create a dictionary that stores this information
        reverse_track_ids = {}
        for batch in range(self.y.shape[0]):
            reverse_track_ids[batch] = {}
        for track in track_ids.keys():
            batch = track_ids[track]['batch']
            label = track_ids[track]['label']
            reverse_track_ids[batch][label] = track

        # Save dictionaries
        self.track_ids = track_ids
        self.reverse_track_ids = reverse_track_ids

        # Identify which tracks have divisions
        self.tracks_with_divisions = []
        for track in self.track_ids.keys():
            if len(self.track_ids[track]['daughters']) > 0:
                self.tracks_with_divisions.append(track)

    def _sub_area(self, X_frame, y_frame, cell_label, num_channels):
        X_padded = np.pad(X_frame, ((self.neighborhood_true_size, self.neighborhood_true_size),
                                    (self.neighborhood_true_size, self.neighborhood_true_size),
                                    (0,0)), mode='constant', constant_values=0)
        y_padded = np.pad(y_frame, ((self.neighborhood_true_size, self.neighborhood_true_size),
                                    (self.neighborhood_true_size, self.neighborhood_true_size),
                                    (0,0)), mode='constant', constant_values=0)
        props = skimage.measure.regionprops(np.int32(y_padded == cell_label))
        center_x, center_y = props[0].centroid
        center_x, center_y = np.int(center_x), np.int(center_y)
        X_reduced = X_padded[
                center_x - self.neighborhood_true_size:center_x + self.neighborhood_true_size,
                center_y - self.neighborhood_true_size:center_y + self.neighborhood_true_size,:]

        # Resize X_reduced in case it is used instead of the neighborhood method
        resize_shape = (2 * self.neighborhood_scale_size + 1,
                        2 * self.neighborhood_scale_size + 1, num_channels)

        # Resize images from bounding box
        X_reduced = resize(X_reduced, resize_shape, mode='constant', preserve_range=True)

        return X_reduced

    def _get_features(self, X, y, frames, labels):
        """
        This function gets the features of a list of cells.
        Cells are defined by lists of frames and labels. The i'th element of
        frames and labels is the frame and label of the i'th cell being grabbed.
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

        neighborhood_shape = (len(frames), 2 * self.neighborhood_scale_size + 1,
                                           2 * self.neighborhood_scale_size + 1, 1)
        future_area_shape = (len(frames) - 1, 2 * self.neighborhood_scale_size + 1,
                                              2 * self.neighborhood_scale_size + 1, 1)

        # Initialize storage for appearances and centroids
        appearances = np.zeros(appearance_shape, dtype=K.floatx())
        centroids = []
        regionprops = []
        neighborhoods = np.zeros(neighborhood_shape, dtype = K.floatx())
        future_areas = np.zeros(future_area_shape, dtype=K.floatx())

        for counter, (frame, cell_label) in enumerate(zip(frames, labels)):
            # Get the bounding box
            X_frame = X[frame] if self.data_format == 'channels_last' else X[:, frame]
            y_frame = y[frame] if self.data_format == 'channels_last' else y[:, frame]

            props = skimage.measure.regionprops(np.int32(y_frame == cell_label))
            minr, minc, maxr, maxc = props[0].bbox
            centroids.append(props[0].centroid)
            regionprops.append(np.array([props[0].area, props[0].perimeter, props[0].eccentricity]))

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

            neighborhoods[counter] = self._sub_area(X_frame, y_frame, cell_label, X.shape[channel_axis])

            if frame != frames[-1]:
                X_future_frame = X[frame + 1] if self.data_format == 'channels_last' else X[:, frame + 1]
                future_areas[counter] = self._sub_area(X_future_frame, y_frame, cell_label, X.shape[channel_axis])

        return [appearances, centroids, neighborhoods, regionprops, future_areas]

    def _create_features(self):
        """
        This function gets the appearances of every cell, crops them out, resizes them,
        and stores them in an matrix. Pre-fetching the appearances should significantly
        speed up the generator. It also gets the centroids and neighborhoods
        """
        number_of_tracks = len(self.track_ids.keys())

        # Initialize the array for the appearances and centroids
        if self.data_format =='channels_first':
            all_appearances_shape = (number_of_tracks, self.x.shape[self.channel_axis],
                                     self.x.shape[self.time_axis], self.crop_dim, self.crop_dim)
        if self.data_format == 'channels_last':
            all_appearances_shape = (number_of_tracks, self.x.shape[self.time_axis],
                                     self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis])
        all_appearances = np.zeros(all_appearances_shape, dtype=K.floatx())

        all_centroids_shape = (number_of_tracks, self.x.shape[self.time_axis], 2)
        all_centroids = np.zeros(all_centroids_shape, dtype=K.floatx())

        all_regionprops_shape = (number_of_tracks, self.x.shape[self.time_axis], 3)
        all_regionprops = np.zeros(all_regionprops_shape, dtype=K.floatx())

        all_neighborhoods_shape = (number_of_tracks, self.x.shape[self.time_axis],
                                     2 * self.neighborhood_scale_size + 1,
                                     2 * self.neighborhood_scale_size + 1, 1)
        all_neighborhoods = np.zeros(all_neighborhoods_shape, dtype=K.floatx())

        all_future_area_shape = (number_of_tracks, self.x.shape[self.time_axis],
                                 2 * self.neighborhood_scale_size + 1,
                                 2 * self.neighborhood_scale_size + 1, 1)
        all_future_areas = np.zeros(all_future_area_shape, dtype=K.floatx())

        for track in self.track_ids.keys():
            batch = self.track_ids[track]['batch']
            label = self.track_ids[track]['label']
            frames = self.track_ids[track]['frames']

            # Make an array of labels that the same length as the frames array
            labels = [label] * len(frames)
            X = self.x[batch]
            y = self.y[batch]

            appearance, centroid, neighborhood, regionprop, future_area = self._get_features(X, y, frames, labels)

            if self.data_format == 'channels_first':
                all_appearances[track,:,np.array(frames),:,:] = appearance
            if self.data_format == 'channels_last':
                all_appearances[track,np.array(frames),:,:,:] = appearance

            all_centroids[track, np.array(frames),:] = centroid
            all_neighborhoods[track, np.array(frames),:,:] = neighborhood
            all_future_areas[track, np.array(frames[:-1]),:,:] = future_area
            all_regionprops[track, np.array(frames),:] = regionprop

        self.all_appearances = all_appearances
        self.all_centroids = all_centroids
        self.all_regionprops = all_regionprops
        self.all_neighborhoods = all_neighborhoods
        self.all_future_areas = all_future_areas

    def _fetch_appearances(self, track, frames):
        """
        This function gets the appearances after they have been
        cropped out of the image
        """
        # TO DO: Check to make sure the frames are acceptable

        if self.data_format == 'channels_first':
            appearances = self.all_appearances[track,:,np.array(frames),:,:]
        if self.data_format == 'channels_last':
            appearances = self.all_appearances[track,np.array(frames),:,:,:]
        return appearances

    def _fetch_centroids(self, track, frames):
        """
        This function gets the centroids after they have been
        extracted and stored
        """
        # TO DO: Check to make sure the frames are acceptable
        return self.all_centroids[track,np.array(frames),:]

    def _fetch_neighborhoods(self, track, frames):
        """
        This function gets the neighborhoods after they have been
        extracted and stored
        """
        # TO DO: Check to make sure the frames are acceptable
        return self.all_neighborhoods[track,np.array(frames),:,:,:]

    def _fetch_future_areas(self, track, frames):
        """
        This function gets the future areas after they have been
        extracted and stored
        """
        # TO DO: Check to make sure the frames are acceptable
        return self.all_future_areas[track,np.array(frames),:,:,:]

    def _fetch_regionprops(self, track, frames):
        """
        This function gets the regionprops after they have been extracted and stored
        """
        # TO DO: Check to make sure the frames are acceptable
        return self.all_regionprops[track,np.array(frames)]

    def _fetch_frames(self, track, division=False):
        """
        This function fetches a random list of frames for a given track.
        If the division flag is true, then the list of frames ends at the cell's
        last appearance if the division flag is true.
        """
        track_id = self.track_ids[track]
        batch = track_id['batch']
        tracked_frames = list(track_id['frames'])

        # We need to have at least one future frame to pick from, so if
        # the last frame of the movie is a tracked frame, remove it
        last_frame = self.x.shape[self.time_axis] - 1
        if last_frame in tracked_frames:
            tracked_frames.remove(last_frame)

        # Get the indices of the tracked_frames list - sometimes frames
        # are skipped
        tracked_frames_index = np.arange(len(tracked_frames))

        # Check if there are enough frames
        enough_frames = len(tracked_frames_index) > self.min_track_length + 1

        # We need to exclude the last frame so that we will always be able to make a comparison
        acceptable_indices = tracked_frames_index[self.min_track_length-1:-1] if enough_frames else tracked_frames_index[:-1]

        # Take the last frame if there is a division, otherwise randomly pick a frame
        index = -1 if division else np.random.choice(acceptable_indices)

        # Select the frames. If there aren't enough frames, repeat the first frame
        # the necessary number of times
        if enough_frames:
            frames = tracked_frames[index+1-self.min_track_length:index+1]
        else:
            frames_temp = tracked_frames[0:index+1]
            missing_frames = self.min_track_length - len(frames_temp)
            frames = [tracked_frames[0]] * missing_frames + frames_temp

        return frames

    def _compute_appearances(self, track_1, frames_1, track_2, frames_2, transform):
        appearance_1 = self._fetch_appearances(track_1, frames_1)
        appearance_2 = self._fetch_appearances(track_2, frames_2)

        # Apply random transforms
        new_appearance_1 = np.zeros(appearance_1.shape, dtype=K.floatx())
        new_appearance_2 = np.zeros(appearance_2.shape, dtype=K.floatx())

        for frame in range(appearance_1.shape[self.time_axis-1]):
            if self.data_format == 'channels_first':
                if transform is not None:
                    app_temp = self.image_data_generator.apply_transform(appearance_1[:,frame,:,:], transform)
                else:
                    app_temp = self.image_data_generator.random_transform(appearance_1[:,frame,:,:])
                app_temp = self.image_data_generator.standardize(app_temp)
                new_appearance_1[:,frame,:,:] = app_temp

            if self.data_format == 'channels_last':
                if transform is not None:
                    app_temp = self.image_data_generator.apply_transform(appearance_1[frame], transform)
                else:
                    self.image_data_generator.random_transform(appearance_1[frame])
                app_temp = self.image_data_generator.standardize(app_temp)
                new_appearance_1[frame] = app_temp

        if self.data_format == 'channels_first':
            if transform is not None:
                app_temp = self.image_data_generator.apply_transform(appearance_2[:,0,:,:], transform)
            else:
                app_temp = self.image_data_generator.random_transform(appearance_2[:,0,:,:])
            app_temp = self.image_data_generator.standardize(app_temp)
            new_appearance_2[:,0,:,:] = app_temp

        if self.data_format == 'channels_last':
            if transform is not None:
                app_temp = self.image_data_generator.apply_transform(appearance_2[0], transform)
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

        distance_1 = distance[0:-1,:]
        distance_2 = distance[-1,:]

        return distance_1, distance_2

    def _compute_regionprops(self, track_1, frames_1, track_2, frames_2, transform):
        regionprop_1 = self._fetch_regionprops(track_1, frames_1)
        regionprop_2 = self._fetch_regionprops(track_2, frames_2)

        return regionprop_1, regionprop_2

    def _compute_neighborhoods(self, track_1, frames_1, track_2, frames_2, transform):
        track_2, frames_2 = None, None # To guarantee we don't use these.
        neighborhood_1 = self._fetch_neighborhoods(track_1, frames_1)
        neighborhood_2 = self._fetch_future_areas(track_1, [frames_1[-1]])

        neighborhoods = np.concatenate([neighborhood_1, neighborhood_2], axis=0)

        for frame in range(neighborhoods.shape[self.time_axis-1]):
            neigh_temp = neighborhoods[frame]
            if transform is not None:
                neigh_temp = self.image_data_generator.apply_transform(neigh_temp, transform)
            else:
                neigh_temp = self.image_data_generator.random_transform(neigh_temp)
            neighborhoods[frame] = neigh_temp

        neighborhood_1 = neighborhoods[0:-1,:,:,:]
        neighborhood_2 = neighborhoods[-1,:,:,:]

        return neighborhood_1, neighborhood_2

    def _compute_feature_shape(self, feature, index_array):
        if feature == "appearance":
            if self.data_format == 'channels_first':
                shape_1 = (len(index_array), self.x.shape[self.channel_axis],
                            self.min_track_length, self.crop_dim, self.crop_dim)
                shape_2 = (len(index_array), self.x.shape[self.channel_axis],
                            self.crop_dim, self.crop_dim)
            else:
                shape_1 = (len(index_array), self.min_track_length,self.crop_dim, self.crop_dim,
                            self.x.shape[self.channel_axis])
                shape_2 = (len(index_array), 1, self.crop_dim, self.crop_dim,
                            self.x.shape[self.channel_axis])

        elif feature == "distance":
            shape_1 = (len(index_array), self.min_track_length, 2)
            shape_2 = (len(index_array), 1, 2)

        elif feature == "neighborhood":
            shape_1 = (len(index_array), self.min_track_length,
                       2 * self.neighborhood_scale_size + 1, 2 * self.neighborhood_scale_size + 1, 1)
            shape_2 = (len(index_array), 1, 2 * self.neighborhood_scale_size + 1,
                       2 * self.neighborhood_scale_size + 1, 1)
        elif feature == "regionprop":
            shape_1 = (len(index_array), self.min_track_length, 3)
            shape_2 = (len(index_array), 1, 3)
        else:
            raise ValueError("_compute_feature_shape: Unknown feature '{}'".format(feature))

        return shape_1, shape_2

    def _compute_feature(self, feature, *args, **kwargs):
        if feature == "appearance":
            return self._compute_appearances(*args, **kwargs)
        elif feature == "distance":
            return self._compute_distances(*args, **kwargs)
        elif feature == "neighborhood":
            return self._compute_neighborhoods(*args, **kwargs)
        elif feature == "regionprop":
            return self._compute_regionprops(*args, **kwargs)
        else:
            raise ValueError("_compute_feature: Unknown feature '{}'".format(feature))

    def _get_batches_of_transformed_samples(self, index_array):
        # Initialize batch_x_1, batch_x_2, and batch_y, as well as cell distance data
        # Compare cells in neighboring frames. Select a sequence of cells/distances 
        # for x1 and 1 cell/distance for x2

        # setup zeroed batch arrays for each feature & batch_y
        batch_features = []
        for feature in self.features:
            shape_1, shape_2 = self._compute_feature_shape(feature, index_array)
            batch_features.append([np.zeros(shape_1, dtype=K.floatx()),
                                   np.zeros(shape_2, dtype=K.floatx())])

        batch_y = np.zeros((len(index_array), 3), dtype=np.int32)

        for i, j in enumerate(index_array):
            # Identify which tracks are going to be selected
            track_id = self.track_ids[j]
            batch = track_id['batch']
            label_1 = track_id['label']

            X = self.x[batch]
            y = self.y[batch]

            # Choose comparison cell
            # Determine what class the track will be - different (0), same (1), division (2)
            division = False
            type_cell = np.random.choice([0, 1, 2], p=[1/3, 1/3, 1/3])

            # Dealing with edge cases
            # If class is division, check if the first cell divides. If not, change tracks
            if type_cell == 2:
                division == True
                if len(track_id['daughters']) == 0:
                    # No divisions so randomly choose a different track that is
                    # guaranteed to have a division
                    new_j = np.random.choice(self.tracks_with_divisions)
                    j = new_j
                    track_id = self.track_ids[j]
                    batch = track_id['batch']
                    label_1 = track_id['label']
                    X = self.x[batch]
                    y = self.y[batch]

            # Get the frames for cell 1 and frames/label for cell 2
            frames_1 = self._fetch_frames(j, division=division)

            # For frame_2, choose the next frame cell 1 appears in
            last_frame_1 = np.amax(frames_1)
            frame_2 = np.amin( [x for x in track_id['frames'] if x > last_frame_1] )
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
                transform = {"theta": self.image_data_generator.rotation_range * np.random.uniform(-1, 1),
                             "flip_horizontal": (np.random.random() < 0.5) if self.image_data_generator.horizontal_flip else False,
                             "flip_vertical": (np.random.random() < 0.5) if self.image_data_generator.vertical_flip else False}

            else:
                transform = None

            for feature_i, feature in enumerate(self.features):
                feature_1, feature_2 = self._compute_feature(feature,
                                                             track_1, frames_1,
                                                             track_2, frames_2,
                                                             transform=transform)
                batch_features[feature_i][0][i] = feature_1
                batch_features[feature_i][1][i] = feature_2

            batch_y[i, type_cell] = 1

        # prepare final batch list
        batch_list = []
        for feature_i, feature in enumerate(self.features):
            batch_feature_1, batch_feature_2 = batch_features[feature_i]
            # Remove singleton dimensions (if min_track_length is 1)
            if self.squeeze:
                if feature == "appearance":
                    batch_feature_1 = np.squeeze(batch_feature_1, axis=self.time_axis)
                    batch_feature_2 = np.squeeze(batch_feature_2, axis=self.time_axis)
                else:
                    batch_feature_1 = np.squeeze(batch_feature_1, axis=1)
                    batch_feature_2 = np.squeeze(batch_feature_2, axis=1)

            batch_list.append(batch_feature_1)
            batch_list.append(batch_feature_2)

        return batch_list, batch_y

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


"""
Bounding box generators adapted from retina net library
"""


class BoundingBoxIterator(Iterator):
    def __init__(self, train_dict, image_data_generator,
                 batch_size=1, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `BoundingBoxIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.y = train_dict['y']

        if self.channel_axis == 3:
            self.num_features = self.y.shape[-1]
        else:
            self.num_features = self.y.shape[1]

        if self.channel_axis == 3:
            self.image_shape = self.x.shape[1:2]
        else:
            self.image_shape = self.x.shape[2:]

        bbox_list = []
        for b in range(self.x.shape[0]):
            for l in range(1, self.num_features - 1):
                if self.channel_axis == 3:
                    mask = self.y[b, :, :, l]
                else:
                    mask = self.y[b, l, :, :]
                props = skimage.measure.regionprops(label(mask))
                bboxes = [np.array(list(prop.bbox) + list(l)) for prop in props]
                bboxes = np.concatenate(bboxes, axis=0)
            bbox_list.append(bboxes)
        self.bbox_list = bbox_list

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(BoundingBoxIterator, self).__init__(self.x.shape[0], batch_size, shuffle, seed)

    def get_annotations(self, y):
        for l in range(1, self.num_features - 1):
            if self.channel_axis == 3:
                mask = y[:, :, l]
            else:
                mask = y[l, :, :]
            props = skimage.measure.regionprops(label(mask))
            bboxes = [np.array(list(prop.bbox) + list(l)) for prop in props]
            bboxes = np.concatenate(bboxes, axis=0)
        return bboxes

    def anchor_targets(self,
                       image_shape,
                       annotations,
                       num_classes,
                       mask_shape=None,
                       negative_overlap=0.4,
                       positive_overlap=0.5,
                       **kwargs):
        return self.anchor_targets_bbox(
            image_shape,
            annotations,
            num_classes,
            mask_shape,
            negative_overlap,
            positive_overlap,
            **kwargs)

    def compute_target(self, annotation):
        labels, annotations, anchors = self.anchor_targets(
            self.image_shape, annotation, self.num_features)
        regression = self.bbox_transform(anchors, annotation)

        # append anchor state to regression targets
        anchor_states = np.max(labels, axis=1, keepdims=True)
        regression = np.append(regression, anchor_states, axis=1)
        return [regression, labels]

    def _get_batches_of_transformed_samples(self, index_array):
        index_array = index_array[0]
        if self.channel_axis == 1:
            batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:4]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:4]))
        else:
            batch_x = np.zeros((len(index_array),
                                self.x.shape[2],
                                self.x.shape[3],
                                self.x.shape[1]))
            if self.y is not None:
                batch_y = np.zeros((len(index_array),
                                    self.y.shape[2],
                                    self.y.shape[3],
                                    self.y.shape[1]))

        regressions_list = []
        labels_list = []

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            if self.channel_axis == 1:
                batch_x[i] = x
                batch_y[i] = y

                # Get the bounding boxes from the transformed masks!
                annotations = self.get_annotations(y)
                regressions, labels = self.compute_target(annotations)
                regressions_list.append(regressions)
                labels_list.append(labels)

            if self.channel_axis == 3:
                raise NotImplementedError('Bounding box generator does not work '
                                          'for channels last yet')

            regressions = np.stack(regressions_list, axis=0)
            labels = np.stack(labels_list, axis=0)

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, [regressions_list, labels_list]

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


"""
RetinaNet and MaskRCNN Generators
"""


class RetinaNetGenerator(_RetinaNetGenerator):

    def __init__(self,
                 direc_name,
                 training_dirs,
                 raw_image_dir,
                 channel_names,
                 annotation_dir,
                 annotation_names,
                 **kwargs):
        self.image_names = []
        self.image_data = {}
        self.image_stack = []
        self.mask_stack = []
        self.base_dir = kwargs.get('base_dir')

        train_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=raw_image_dir,
            channel_names=channel_names)

        annotation_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=annotation_dir,
            channel_names=annotation_names)

        self.image_stack = self.generate_subimage(train_files, 3, 3, True)
        self.mask_stack = self.generate_subimage(annotation_files, 3, 3, False)

        self.classes = {'cell': 0}

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = self._read_annotations(self.mask_stack)
        self.image_names = list(self.image_data.keys())
        super(RetinaNetGenerator, self).__init__(**kwargs)

    def list_file_deepcell(self, dir_name, training_dirs, image_dir, channel_names):
        """
        List all image files inside each `dir_name/training_dir/image_dir`
        with "channel_name" in the filename.
        """
        filelist = []
        for direc in training_dirs:
            imglist = os.listdir(os.path.join(dir_name, direc, image_dir))

            for channel in channel_names:
                for img in imglist:
                    # if channel string is NOT in image file name, skip it.
                    if not fnmatch(img, '*{}*'.format(channel)):
                        continue
                    image_file = os.path.join(dir_name, direc, image_dir, img)
                    filelist.append(image_file)
        return sorted(filelist)

    def _read_annotations(self, masks_list):
        result = {}
        for cnt, image in enumerate(masks_list):
            result[cnt] = []
            p = skimage.measure.regionprops(label(image))

            cell_count = 0
            for index in range(len(np.unique(label(image))) - 1):
                y1, x1, y2, x2 = p[index].bbox
                result[cnt].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
                cell_count += 1
            if cell_count == 0:
                logging.warning('No cells found in image {}'.format(cnt))
        return result

    def generate_subimage(self, img_pathstack, horizontal, vertical, flag):
        sub_img = []
        for img_path in img_pathstack:
            img = np.asarray(np.float32(imread(img_path)))
            if flag:
                img = (img / np.max(img))
            vway = np.zeros(vertical + 1)  # The dimentions of vertical cuts
            hway = np.zeros(horizontal + 1)  # The dimentions of horizontal cuts
            vcnt = 0  # The initial value for vertical
            hcnt = 0  # The initial value for horizontal

            for i in range(vertical + 1):
                vway[i] = int(vcnt)
                vcnt += (img.shape[1] / vertical)

            for j in range(horizontal + 1):
                hway[j] = int(hcnt)
                hcnt += (img.shape[0] / horizontal)

            vb = 0

            for i in range(len(hway) - 1):
                for j in range(len(vway) - 1):
                    vb += 1

            for i in range(len(hway) - 1):
                for j in range(len(vway) - 1):
                    s = img[int(hway[i]):int(hway[i + 1]), int(vway[j]):int(vway[j + 1])]
                    sub_img.append(s)

        if flag:
            sub_img = [np.tile(np.expand_dims(i, axis=-1), (1, 1, 3)) for i in sub_img]

        return sub_img

    def size(self):
        """Size of the dataset."""
        return len(self.image_names)

    def num_classes(self):
        """Number of classes in the dataset."""
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """Map name to label."""
        return self.classes[name]

    def label_to_name(self, label):
        """Map label to name."""
        return self.labels[label]

    def image_path(self, image_index):
        """Returns the image path for image_index."""
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """Compute the aspect ratio for an image with image_index."""
        image = self.image_stack[image_index]
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        """Load an image at the image_index."""
        return self.image_stack[image_index]

    def load_annotations(self, image_index):
        """Load annotations for an image_index."""
        path = self.image_names[image_index]
        annots = self.image_data[path]
        boxes = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = 'cell'
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes


class MaskRCNNGenerator(_MaskRCNNGenerator):
    def __init__(self,
                 direc_name,
                 training_dirs,
                 raw_image_dir,
                 channel_names,
                 annotation_dir,
                 annotation_names,
                 base_dir=None,
                 image_min_side=200,
                 image_max_side=200,
                 crop_iterations=1,
                 **kwargs):
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        self.image_stack = []

        train_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=raw_image_dir,
            channel_names=channel_names)

        annotation_files = self.list_file_deepcell(
            dir_name=direc_name,
            training_dirs=training_dirs,
            image_dir=annotation_dir,
            channel_names=annotation_names)

        store = self.randomcrops(
            train_files,
            annotation_files,
            image_min_side,
            image_max_side,
            iteration=crop_iterations)

        self.image_stack = store[0]
        self.classes = {'cell': 0}

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = self._read_annotations(store[1])

        self.image_names = list(self.image_data.keys())

        # Override default Generator value with custom anchor_targets_bbox
        if 'compute_anchor_targets' not in kwargs:
            kwargs['compute_anchor_targets'] = anchor_targets_bbox

        super(MaskRCNNGenerator, self).__init__(
            image_min_side=image_min_side,
            image_max_side=image_max_side,
            **kwargs)

    def list_file_deepcell(self, dir_name, training_dirs, image_dir, channel_names):
        """
        List all image files inside each `dir_name/training_dir/image_dir`
        with "channel_name" in the filename.

        """
        filelist = []
        for direc in training_dirs:
            imglist = os.listdir(os.path.join(dir_name, direc, image_dir))

            for channel in channel_names:
                for img in imglist:
                    # if channel string is NOT in image file name, skip it.
                    if not fnmatch(img, '*{}*'.format(channel)):
                        continue
                    image_file = os.path.join(dir_name, direc, image_dir, img)
                    filelist.append(image_file)
        return sorted(filelist)

    def randomcrops(self, dirpaths, maskpaths, size_x, size_y, iteration=1):
        img = cv2.imread(dirpaths[0], 0)
        img_y = img.shape[0]
        img_x = img.shape[1]
        act_x = img_x - size_x
        act_y = img_y - size_y
        if act_x < 0 or act_y < 0:
            logging.warning('Image to crop is of a smaller size')
            return ([], [])
        outputi = []
        outputm = []
        while iteration > 0:
            cropindex = []
            for path in dirpaths:
                rand_x = np.random.randint(0, act_x)
                rand_y = np.random.randint(0, act_y)
                cropindex.append((rand_x, rand_y))
                image = cv2.imread(path, 0)
                newimg = image[rand_y:rand_y + size_y, rand_x:rand_x + size_x]
                newimg = np.tile(np.expand_dims(newimg, axis=-1), (1, 1, 3))
                outputi.append(newimg)

            for i, path in enumerate(maskpaths):
                image = cv2.imread(path, 0)
                rand_x = cropindex[i][0]
                rand_y = cropindex[i][1]
                newimg = image[rand_y:rand_y + size_y, rand_x:rand_x + size_x]
                outputm.append(newimg)

            iteration -= 1
        return (outputi, outputm)

    def _read_annotations(self, maskarr):
        result = {}
        for cnt, image in enumerate(maskarr):
            result[cnt] = []
            l = label(image)
            p = skimage.measure.regionprops(l)
            cell_count = 0
            for index in range(len(np.unique(l)) - 1):
                y1, x1, y2, x2 = p[index].bbox
                result[cnt].append({
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                    'class': 'cell',
                    'mask_path': np.where(l == index + 1, 1, 0)
                })
                cell_count += 1
            print('Image number {} has {} cells'.format(cnt, cell_count))
            # If there are no cells in this image, remove it from the annotations
            if not result[cnt]:
                del result[cnt]
        return result

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        # image = Image.open(self.image_path(image_index))
        # return float(image.width) / float(image.height)
        image = self.image_stack[image_index]
        return float(image.shape[1]) / float(image.shape[0])

    def load_image(self, image_index):
        # return read_image_bgr(self.image_path(image_index))
        return self.image_stack[image_index]

    def load_annotations(self, image_index):
        path = self.image_names[image_index]
        annots = self.image_data[path]

        # find mask size in order to allocate the right dimension for the annotations
        annotations = np.zeros((len(annots), 5))
        masks = []

        for idx, annot in enumerate(annots):
            annotations[idx, 0] = float(annot['x1'])
            annotations[idx, 1] = float(annot['y1'])
            annotations[idx, 2] = float(annot['x2'])
            annotations[idx, 3] = float(annot['y2'])
            annotations[idx, 4] = self.name_to_label(annot['class'])
            mask = annot['mask_path']
            mask = (mask > 0).astype(np.uint8)  # convert from 0-255 to binary mask
            masks.append(np.expand_dims(mask, axis=-1))

        return annotations, masks
