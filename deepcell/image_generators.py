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
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage.io import imread

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

try:
    from tensorflow.python.keras.utils import conv_utils
except ImportError:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras._impl.keras.utils import conv_utils

# Check if ImageDataGenerator is 1.11.0 or later
if not hasattr(ImageDataGenerator, 'apply_transform'):
    # tf.version is 1.10.0 or earlier, use keras_preprocessing classes
    from keras_preprocessing.image import Iterator
    from keras_preprocessing.image import ImageDataGenerator

from keras_retinanet.preprocessing.generator import Generator as _RetinaNetGenerator
from keras_maskrcnn.preprocessing.generator import Generator as _MaskRCNNGenerator

from deepcell.utils.data_utils import sample_label_movie
from deepcell.utils.data_utils import sample_label_matrix
from deepcell.utils.transform_utils import deepcell_transform
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.utils.transform_utils import distance_transform_3d
from deepcell.utils.retinanet_anchor_utils import anchor_targets_bbox
from deepcell.utils.retinanet_anchor_utils import anchors_for_shape
from deepcell.utils.retinanet_anchor_utils import guess_shapes


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

    Args:
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
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))
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

    Args:
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

        Args:
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

    Args:
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
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))
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

    Args:
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

        Args:
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

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x: 3D tensor, single image.
            y: 3D tensor, label mask for `x`, optional.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
            If `y` is passed, it is transformed if necessary and returned.
        """
        params = self.get_random_transform(x.shape, seed)
        x = self.apply_transform(x, params)

        if y is None:
            return x

        # Nullify the transforms that don't affect `y`
        params['brightness'] = None
        params['channel_shift_intensity'] = None
        y = self.apply_transform(y, params)
        return x, y


class MovieDataGenerator(ImageDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Args:
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

        Args:
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
                                '`featurewise_std_normalization`, but it hasn\'t '
                                'been fit on any training data. Fit it '
                                'first by calling `.fit(numpy_data)`.')
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

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x: 4D tensor, stack of images.
            y: 4D tensor, label mask for `x`, optional.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
            If `y` is passed, it is transformed if necessary and returned.
        """
        # Note: Workaround to use self.apply_transform on our 4D tensor
        self.row_axis -= 1
        self.col_axis -= 1
        self.time_axis -= 1
        self.channel_axis -= 1
        x_new = np.empty(x.shape)
        if y is not None:
            y_new = np.empty(y.shape)
        # apply_transform expects ndim=3, but we are ndim=4
        for frame in range(x.shape[self.time_axis]):
            if self.data_format == 'channels_first':
                params = self.get_random_transform(x[:, frame].shape, seed)
                x_trans = self.apply_transform(x[:, frame], params)
                x_new[:, frame] = np.rollaxis(x_trans, -1, 0)
            else:
                params = self.get_random_transform(x[frame].shape, seed)
                x_new[frame] = self.apply_transform(x[frame], params)

            if y is not None:
                params['brightness'] = None
                params['channel_shift_intensity'] = None
                if self.data_format == 'channels_first':
                    y_trans = self.apply_transform(y[:, frame], params)
                    y_new[:, frame] = np.rollaxis(y_trans, 1, 0)
                else:
                    y_new[frame] = self.apply_transform(y[frame], params)
        # Note: Undo workaround
        self.row_axis += 1
        self.col_axis += 1
        self.time_axis += 1
        self.channel_axis += 1
        if y is None:
            return x_new
        return x_new, y_new

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
                x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = scipy.linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s[np.newaxis] + self.zca_epsilon)
            self.principal_components = (u * s_inv).dot(u.T)


class MovieArrayIterator(Iterator):
    """Iterator yielding data from two 5D Numpy arrays (`X and `y`).

    Args:
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
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError('`X` (movie data) and `y` (labels) '
                             'should have the same size. Found '
                             'Found x.shape = {}, y.shape = {}'.format(
                                 X.shape, y.shape))

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
                x = self.x[j, time_start:time_end, ...]
                if self.y is not None:
                    y = self.y[j, time_start:time_end, ...]
            elif self.time_axis == 2:
                x = self.x[j, :, time_start:time_end, ...]
                if self.y is not None:
                    y = self.y[j, :, time_start:time_end, ...]

            if self.y is not None:
                x, y = self.movie_data_generator.random_transform(
                    x.astype(K.floatx()), y=y)
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

    Args:
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
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError('`X` (movie data) and `y` (labels) '
                             'should have the same size. Found '
                             'Found x.shape = {}, y.shape = {}'.format(
                                 X.shape, y.shape))
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

    Args:
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

        Args:
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


class SiameseDataGenerator(ImageDataGenerator):
    def flow(self,
             train_dict,
             crop_dim=14,
             min_track_length=5,
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
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class SiameseIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 crop_dim=14,
                 min_track_length=5,
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
        self.crop_dim = crop_dim
        self.min_track_length = min_track_length
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.track_ids = self._get_track_ids()

        super(SiameseIterator, self).__init__(
            len(self.track_ids), batch_size, shuffle, seed)

    def _get_track_ids(self):
        """
        This function builds the track id's. It returns a dictionary that
        contains the batch number and label number of each each track.
        Creates unique cell IDs, as cell labels are NOT unique across batches.

        """
        track_counter = 0
        track_ids = {}
        for batch in range(self.y.shape[0]):
            y_batch = self.y[batch]
            num_cells = np.amax(y_batch)
            for cell in range(1, num_cells + 1):
                # count number of pixels cell occupies in each frame
                y_true = np.sum(y_batch == cell, axis=(self.row_axis - 1, self.col_axis - 1))
                # get indices of frames where cell is present
                y_index = np.where(y_true > 0)[0]
                if y_index.size > 0:  # if cell is present at all
                    start_frame = np.amin(y_index)
                    stop_frame = np.amax(y_index)
                    track_ids[track_counter] = {
                        'batch': batch,
                        'label': cell,
                        'frames': y_index
                    }
                    track_counter += 1
        return track_ids

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            batch_shape = (len(index_array),
                           self.x.shape[self.channel_axis],
                           self.crop_dim,
                           self.crop_dim)
        else:
            batch_shape = (len(index_array),
                           self.crop_dim,
                           self.crop_dim,
                           self.x.shape[self.channel_axis])

        batch_x_1 = np.zeros(batch_shape, dtype=K.floatx())
        batch_x_2 = np.zeros(batch_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array), 2), dtype='int32')

        for i, j in enumerate(index_array):
            # Identify which tracks are going to be selected
            track_id = self.track_ids[j]
            batch = track_id['batch']
            label_1 = track_id['label']
            tracked_frames = track_id['frames']
            frame_1 = np.random.choice(tracked_frames)  # Select a frame from the track

            X = self.x[batch]
            y = self.y[batch]

            # Choose comparison cell
            # Determine what class the track will be - different (0), same (1)
            is_same_cell = np.random.random_integers(0, 1)

            # Select another frame from the same track
            if is_same_cell:
                label_2 = label_1
                frame_2 = np.random.choice(track_id['frames'])

            # Select another frame from a different track
            if not is_same_cell:
                # all_labels = np.arange(1, np.amax(y) + 1)
                all_labels = np.delete(np.unique(y), 0)  # all labels in y but 0 (background)
                acceptable_labels = np.delete(all_labels, np.where(all_labels == label_1))
                is_valid_label = False
                while not is_valid_label:
                    # get a random cell label from our acceptable list
                    label_2 = np.random.choice(acceptable_labels)

                    # count number of pixels cell occupies in each frame
                    y_true = np.sum(y == label_2, axis=(
                        self.row_axis - 1, self.col_axis - 1, self.channel_axis - 1))

                    y_index = np.where(y_true > 0)[0]  # get frames where cell is present
                    is_valid_label = y_index.any()  # label_2 is in a frame
                    if not is_valid_label:
                        # remove invalid label from list of acceptable labels
                        acceptable_labels = np.delete(
                            acceptable_labels, np.where(acceptable_labels == label_2))

                frame_2 = np.random.choice(y_index)  # get random frame with label_2

            # Get appearances
            frames = [frame_1, frame_2]
            labels = [label_1, label_2]

            appearances = self._get_appearances(X, y, frames, labels)
            if self.data_format == 'channels_first':
                appearances = [appearances[:, 0], appearances[:, 1]]
            else:
                appearances = [appearances[0], appearances[1]]

            # Apply random transformations
            for k, appearance in enumerate(appearances):
                appearance = self.image_data_generator.random_transform(appearance)
                appearance = self.image_data_generator.standardize(appearance)
                appearances[k] = appearance

            batch_x_1[i] = appearances[0]
            batch_x_2[i] = appearances[1]
            batch_y[i, is_same_cell] = 1

        return [batch_x_1, batch_x_2], batch_y

    def _get_appearances(self, X, y, frames, labels):
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
        appearances = np.zeros(appearance_shape, dtype=K.floatx())
        for counter, (frame, cell_label) in enumerate(zip(frames, labels)):
            # Get the bounding box
            y_frame = y[frame] if self.data_format == 'channels_last' else y[:, frame]
            props = regionprops(np.int32(y_frame == cell_label))
            minr, minc, maxr, maxc = props[0].bbox

            # Extract images from bounding boxes
            if self.data_format == 'channels_first':
                appearance = X[:, frame, minr:maxr, minc:maxc]
                resize_shape = (X.shape[channel_axis], self.crop_dim, self.crop_dim)
            else:
                appearance = X[frame, minr:maxr, minc:maxc, :]
                resize_shape = (self.crop_dim, self.crop_dim, X.shape[channel_axis])

            # Resize images from bounding box
            max_value = np.amax([np.amax(appearance), np.absolute(np.amin(appearance))])
            appearance /= max_value
            appearance = resize(appearance, resize_shape)
            appearance *= max_value
            if self.data_format == 'channels_first':
                appearances[:, counter] = appearance
            else:
                appearances[counter] = appearance

        return appearances

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


class RetinaNetGenerator(ImageFullyConvDataGenerator):
    """Generates batches of tensor image data with real-time data augmentation.
    The data will be looped over (in batches).

    Args:
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
             compute_shapes=guess_shapes,
             batch_size=32,
             shuffle=False,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
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
        return RetinaNetIterator(
            train_dict,
            self,
            compute_shapes=compute_shapes,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class RetinaNetIterator(Iterator):
    """Iterator yielding data from Numpy arrayss (`X and `y`).

    Args:
        train_dict: dictionary consisting of numpy arrays for `X` and `y`.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        compute_shapes: functor for generating shapes, based on the model.
        min_objects: Integer, image with fewer than `min_objects` are ignored.
        num_classes: Integer, number of classes for classification.
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
                 compute_shapes=guess_shapes,
                 min_objects=3,
                 num_classes=1,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        X, y = train_dict['X'], train_dict['y']
        if X.shape[0] != y.shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 X.shape, y.shape))

        if X.ndim != 4:
            raise ValueError('Input data in `RetinaNetIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', X.shape)

        self.x = np.asarray(X, dtype=K.floatx())
        self.y = np.asarray(y, dtype='int32')

        # `compute_shapes` changes based on the model backbone.
        self.compute_shapes = compute_shapes
        self.min_objects = min_objects
        self.num_classes = num_classes
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        invalid_batches = []
        # Remove images with small numbers of cells
        for b in range(self.x.shape[0]):
            y_batch = np.squeeze(self.y[b], axis=self.channel_axis - 1)
            y_batch = clear_border(y_batch)
            y_batch = np.expand_dims(y_batch, axis=self.channel_axis - 1)

            self.y[b] = y_batch

            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s objects',
                            invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.y = np.delete(self.y, invalid_batches, axis=0)
        self.x = np.delete(self.x, invalid_batches, axis=0)

        super(RetinaNetIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def filter_annotations(self, image, annotations):
        """Filter annotations by removing those that are outside of the
        image bounds or whose width/height < 0.

        Args:
            image: ndarray, the raw image data.
            annotations: dict of annotations including `labels` and `bboxes`
        """
        row_axis = 1 if self.data_format == 'channels_first' else 0
        invalid_indices = np.where(
            (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
            (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
            (annotations['bboxes'][:, 0] < 0) |
            (annotations['bboxes'][:, 1] < 0) |
            (annotations['bboxes'][:, 2] > image.shape[row_axis]) |
            (annotations['bboxes'][:, 3] > image.shape[row_axis + 1])
        )[0]

        # delete invalid indices
        if invalid_indices.size > 0:
            logging.warn('Image with shape {} contains the following invalid '
                         'boxes: {}.'.format(
                             image.shape,
                             annotations['bboxes'][invalid_indices, :]))

            for k in annotations.keys():
                filtered = np.delete(annotations[k], invalid_indices, axis=0)
                annotations[k] = filtered
        return annotations

    def load_annotations(self, y):
        """Generate bounding box and label annotations for a tensor

        Args:
            y: tensor to annotate

        Returns:
            annotations: dict of `bboxes` and `labels`
        """
        labels, bboxes = [], []
        for prop in regionprops(np.squeeze(y.astype('int'))):
            bboxes.append(prop.bbox)
            labels.append(0)  # boolean object detection

        annotations = {'labels': np.array(labels), 'bboxes': np.array(bboxes)}
        annotations = self.filter_annotations(y, annotations)
        return annotations

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))

        annotations_list = []

        for i, j in enumerate(index_array):
            x = self.x[j]
            y = self.y[j]

            x, y = self.image_data_generator.random_transform(x, y)

            # Get the bounding boxes from the transformed masks!
            annotations = self.load_annotations(y)
            annotations_list.append(annotations)

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

        anchors = anchors_for_shape(
            batch_x.shape[1:],
            anchor_params=None,
            shapes_callback=self.compute_shapes)

        regressions, labels = anchor_targets_bbox(
            anchors,
            batch_x,
            annotations_list,
            self.num_classes)

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img_x = np.expand_dims(batch_x[i, ..., 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, [regressions, labels]

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


class ShvmRetinaNetGenerator(_RetinaNetGenerator):

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
        super(ShvmRetinaNetGenerator, self).__init__(**kwargs)

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
            p = regionprops(label(image))

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
            p = regionprops(l)
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
