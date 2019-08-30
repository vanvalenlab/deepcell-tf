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

import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.transform import resize

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

from deepcell.utils.data_utils import sample_label_movie
from deepcell.utils.data_utils import sample_label_matrix
from deepcell.utils.transform_utils import pixelwise_transform
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.utils.transform_utils import distance_transform_3d
from deepcell.utils.retinanet_anchor_utils import anchor_targets_bbox
from deepcell.utils.retinanet_anchor_utils import anchors_for_shape
from deepcell.utils.retinanet_anchor_utils import guess_shapes


def _transform_masks(y, transform, data_format=None, **kwargs):
    """Based on the transform key, apply a transform function to the masks.

    More detailed description. Caution for unknown transform keys.

    Args:
        y: labels of ndim 4 or 5
        transform: one of {"deepcell", "disc", "watershed", "centroid", None}

    Returns:
        y_transform: the output of the given transform function on y

    Raises:
        IOError: An error occurred
    """
    valid_transforms = {
        'pixelwise',
        'disc',
        'watershed',
        'centroid',
        'fgbg'
    }

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
        if transform == 'deepcell':
            raise ValueError('`deepcell` transform has been replaced with the '
                             '`pixelwise` transform.')
        if transform not in valid_transforms:
            raise ValueError('`{}` is not a valid transform'.format(transform))

    if transform == 'pixelwise':
        dilation_radius = kwargs.pop('dilation_radius', None)
        separate_edge_classes = kwargs.pop('separate_edge_classes', False)
        y_transform = pixelwise_transform(y, dilation_radius, data_format=data_format,
                                          separate_edge_classes=separate_edge_classes)

    elif transform == 'watershed':
        distance_bins = kwargs.pop('distance_bins', 4)
        erosion = kwargs.pop('erosion_width', 0)

        if data_format == 'channels_first':
            y_transform = np.zeros(tuple([y.shape[0]] + list(y.shape[2:])))
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
    Sampling will generate a window_size image classifying the center pixel,

    Args:
        train_dict: dictionary consisting of numpy arrays for X and y.
        image_data_generator: Instance of ImageDataGenerator
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        window_size: size of sampling window around each pixel
        balance_classes: balance class representation when sampling
        max_class_samples: maximum number of samples per class.
        seed: Random seed for data shuffling.
        data_format: String, one of 'channels_first', 'channels_last'.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if save_to_dir is set).
        save_format: Format to use for saving sample images
            (if save_to_dir is set).
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

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1],
              while with width_shift_range=1.0 possible values are floats in
              the interval [-1.0, +1.0).

        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:

                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval: float or int, value used for points outside the boundaries
            when fill_mode = "constant".
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

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

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
            train_dict: dictionary consisting of numpy arrays for X and y.
            image_data_generator: Instance of ImageDataGenerator
                to use for random transformations and normalization.
            batch_size: Integer, size of a batch.
            shuffle: Boolean, whether to shuffle the data between epochs.
            window_size: size of sampling window around each pixel
            balance_classes: balance class representation when sampling
            max_class_samples: maximum number of samples per class.
            seed: Random seed for data shuffling.
            data_format: String, one of 'channels_first', 'channels_last'.
            save_to_dir: Optional directory where to save the pictures
                being yielded, in a viewable format. This is useful
                for visualizing the random transformations being
                applied, for debugging purposes.
            save_prefix: String prefix to use for saving sample
                images (if save_to_dir is set).
            save_format: Format to use for saving sample images
                (if save_to_dir is set).
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
    """Iterator yielding data from Numpy arrayss (X and y).

    Args:
        train_dict: dictionary consisting of numpy arrays for X and y.
        image_data_generator: Instance of ImageDataGenerator
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of 'channels_first', 'channels_last'.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if save_to_dir is set).
        save_format: Format to use for saving sample images
            (if save_to_dir is set).
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

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1], while with
              width_shift_range=1.0 possible values are floats in the interval
              [-1.0, +1.0).

        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval: float or int, value used for points outside the boundaries
            when fill_mode = "constant".
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

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

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
            save_prefix: str (default: ''). Prefix to use for filenames of
                saved pictures (only relevant if save_to_dir is set).
            save_format: one of "png", "jpeg". Default: "png".
                (only relevant if save_to_dir is set)

        Returns:
            An Iterator yielding tuples of (x, y) where x is a numpy array
            of image data and y is a numpy array of labels of the same shape.
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
            x: 3D tensor or list of 3D tensors,
                single image.
            y: 3D tensor or list of 3D tensors,
                label mask(s) for x, optional.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
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
            y = [self.apply_transform(y_i, params) for y_i in y]
        else:
            y = self.apply_transform(y, params)

        self.interpolation_order = _interpolation_order
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

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1],
              while with width_shift_range=1.0 possible values are floats in
              the interval [-1.0, +1.0).

        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval: float or int, value used for points outside the boundaries
            when fill_mode = "constant".
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

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

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
            save_prefix: str (default: ''). Prefix to use for filenames of
                saved pictures (only relevant if save_to_dir is set).
            save_format: one of "png", "jpeg". Default: "png".
                (only relevant if save_to_dir is set)

        Returns:
            An Iterator yielding tuples of (x, y) where x is a numpy array
            of image data and y is a numpy array of labels of the same shape.
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
            y: 4D tensor, label mask for x, optional.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
            If y is passed, it is transformed if necessary and returned.
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
                _interpolation_order = self.interpolation_order
                self.interpolation_order = 0
                if self.data_format == 'channels_first':
                    y_trans = self.apply_transform(y[:, frame], params)
                    y_new[:, frame] = np.rollaxis(y_trans, 1, 0)
                else:
                    y_new[frame] = self.apply_transform(y[frame], params)
                self.interpolation_order = _interpolation_order
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
            rounds: If augment,
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
    """Iterator yielding data from two 5D Numpy arrays (X and y).

    Args:
        train_dict: dictionary consisting of numpy arrays for X and y.
        movie_data_generator: Instance of MovieDataGenerator
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        frames_per_batch: size of z axis in generated batches
        seed: Random seed for data shuffling.
        data_format: String, one of 'channels_first', 'channels_last'.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if save_to_dir is set).
        save_format: Format to use for saving sample images
            (if save_to_dir is set).
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
    """Iterator yielding data from two 5D Numpy arrays (X and y).

    Sampling will generate a window_size voxel classifying the center pixel,

    Args:
        train_dict: dictionary consisting of numpy arrays for X and y.
        movie_data_generator: Instance of MovieDataGenerator
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        window_size: size of sampling window around each pixel
        balance_classes: balance class representation when sampling
        max_class_samples: maximum number of samples per class.
        seed: Random seed for data shuffling.
        data_format: String, one of 'channels_first', 'channels_last'.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if save_to_dir is set).
        save_format: Format to use for saving sample images
            (if save_to_dir is set).
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

            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
              (-width_shift_range, +width_shift_range)
            - With width_shift_range=2 possible values are ints [-1, 0, +1],
              same as with width_shift_range=[-1, 0, +1],
              while with width_shift_range=1.0 possible values are floats in
              the interval [-1.0, +1.0).

        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.

            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                - 'nearest':  aaaaaaaa|abcd|dddddddd
                - 'reflect':  abcddcba|abcd|dcbaabcd
                - 'wrap':  abcdabcd|abcd|abcdabcd

        cval: float or int, value used for points outside the boundaries
            when fill_mode = "constant".
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

            - "channels_last" mode means that the images should have shape
              (samples, height, width, channels),
            - "channels_first" mode means that the images should have shape
              (samples, channels, height, width).
            - It defaults to the image_data_format value found in your
              Keras config file at "~/.keras/keras.json".
            - If you never set it, then it will be "channels_last".

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
            save_prefix: str (default: ''). Prefix to use for filenames of
                saved pictures (only relevant if save_to_dir is set).
            save_format: one of "png", "jpeg". Default: "png".
                (only relevant if save_to_dir is set)

        Returns:
            An Iterator yielding tuples of (x, y) where x is a numpy array
            of image data and y is a numpy array of labels of the same shape.
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
Custom tracking generators
"""


class SiameseDataGenerator(ImageDataGenerator):
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
                (-width_shift_range, +width_shift_range)
            With width_shift_range=2 possible values are ints [-1, 0, +1],
            same as with width_shift_range=[-1, 0, +1],
            while with width_shift_range=1.0 possible values are floats in
            the interval [-1.0, +1.0).
        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: float or int, value used for points outside the boundaries
            when fill_mode = "constant".
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
                (samples, height, width, channels),
            "channels_first" mode means that the images should have shape
                (samples, channels, height, width).
            It defaults to the image_data_format value found in your
                Keras config file at "~/.keras/keras.json".
            If you never set it, then it will be "channels_last".
        validation_split: float, fraction of images reserved for validation
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
    """Iterator yielding two sets of features (X) and the relationship (y)
    Features are passed in as a list of feature names, while the y is one of:
        "same", "different", or "daughter"

    Arguments:
        train_dict: dictionary consisting of numpy arrays for X and y.
        image_data_generator: Instance of ImageDataGenerator
            to use for random transformations and normalization.
        features: List of Strings, feature names to calculate and yield.
        crop_dim: Integer, size of the resized appearance images
        min_track_length: Integer, minimum number of frames to track over.
        neighborhood_scale_size: Integer, size of resized neighborhood images
        neighborhood_true_size: Integer, size of cropped neighborhood images
        sync_transform: Boolean, whether to transform the features.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of 'channels_first', 'channels_last'.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if save_to_dir is set).
        save_format: Format to use for saving sample images
            (if save_to_dir is set).
    """
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 features,
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
        self._create_track_ids()
        self._create_features()

        super(SiameseIterator, self).__init__(
            len(self.track_ids), batch_size, shuffle, seed)

    def _remove_bad_images(self):
        """Iterate over all image batches and remove images with only one cell.
        """
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
            A dict containing the batch and label of each each track.
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
            X_frame = np.rollaxis(X_frame, -1, 0)
            y_frame = np.rollaxis(y_frame, -1, 0)

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

        neighborhood_shape = (len(frames),
                              2 * self.neighborhood_scale_size + 1,
                              2 * self.neighborhood_scale_size + 1,
                              1)

        # future area should not include last frame in movie
        last_frame = self.x.shape[self.time_axis] - 1
        if last_frame in frames:
            future_area_len = len(frames) - 1
        else:
            future_area_len = len(frames)

        future_area_shape = (future_area_len,
                             2 * self.neighborhood_scale_size + 1,
                             2 * self.neighborhood_scale_size + 1,
                             1)

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

        all_neighborhoods_shape = (number_of_tracks,
                                   self.x.shape[self.time_axis],
                                   2 * self.neighborhood_scale_size + 1,
                                   2 * self.neighborhood_scale_size + 1,
                                   1)
        all_neighborhoods = np.zeros(all_neighborhoods_shape, dtype=K.floatx())

        all_future_area_shape = (number_of_tracks,
                                 self.x.shape[self.time_axis] - 1,
                                 2 * self.neighborhood_scale_size + 1,
                                 2 * self.neighborhood_scale_size + 1,
                                 1)
        all_future_areas = np.zeros(all_future_area_shape, dtype=K.floatx())

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
                appearance = np.transpose(appearance, (1, 2, 3, 0))
                all_appearances = np.transpose(all_appearances, (0, 2, 3, 4, 1))

            all_appearances[track, np.array(frames), :, :, :] = appearance

            if self.data_format == 'channels_first':
                all_appearances = np.transpose(all_appearances, (0, 4, 1, 2, 3))

            all_centroids[track, np.array(frames), :] = centroid
            all_neighborhoods[track, np.array(frames), :, :] = neighborhood

            # future area should never include last frame
            last_frame = self.x.shape[self.time_axis] - 1
            if last_frame in frames:
                frames_without_last = [f for f in frames if f != last_frame]
                all_future_areas[track, np.array(frames_without_last), :, :] = future_area
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
            return self.all_appearances[track, :, np.array(frames), :, :]
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
            return self.all_neighborhoods[track, :, np.array(frames), :, :]
        return self.all_neighborhoods[track, np.array(frames), :, :, :]

    def _fetch_future_areas(self, track, frames):
        """Gets the future areas after they have been extracted and stored
        """
        # TODO: Check to make sure the frames are acceptable
        if self.data_format == 'channels_first':
            return self.all_future_areas[track, :, np.array(frames), :, :]
        return self.all_future_areas[track, np.array(frames), :, :, :]

    def _fetch_regionprops(self, track, frames):
        """Gets the regionprops after they have been extracted and stored
        """
        # TODO: Check to make sure the frames are acceptable
        return self.all_regionprops[track, np.array(frames)]

    def _fetch_frames(self, track, division=False):
        """Fetch a random interval of frames given a track:

           If division, grab the last min_track_length frames.
           Otherwise, grab any interval of frames of length min_track_length
           that does not include the last tracked frame.

           Args:
               track: integer, used to look up track ID
               division: boolean, is the event being tracked a division

           Returns:
               list of interval of frames of length min_track_length
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

        neighborhoods = np.concatenate([neighborhood_1, neighborhood_2], axis=0)

        for frame in range(neighborhoods.shape[self.time_axis - 1]):
            neigh_temp = neighborhoods[frame]
            if transform is not None:
                neigh_temp = self.image_data_generator.apply_transform(
                    neigh_temp, transform)
            else:
                neigh_temp = self.image_data_generator.random_transform(neigh_temp)
            neighborhoods[frame] = neigh_temp

        neighborhood_1 = neighborhoods[0:-1, :, :, :]
        neighborhood_2 = neighborhoods[-1, :, :, :]

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
            shape_1 = (len(index_array),
                       self.min_track_length,
                       2 * self.neighborhood_scale_size + 1,
                       2 * self.neighborhood_scale_size + 1,
                       1)
            shape_2 = (len(index_array),
                       1,
                       2 * self.neighborhood_scale_size + 1,
                       2 * self.neighborhood_scale_size + 1,
                       1)
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
            # Oversample division case to improve division accuracy
            # type_cell = np.random.choice([0, 1, 2], p=[3 / 10, 3 / 10, 4 / 10])

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

        # prepare final batch list
        batch_list = []
        for feature_i, feature in enumerate(self.features):
            batch_feature_1, batch_feature_2 = batch_features[feature_i]
            # Remove singleton dimensions (if min_track_length is 1)
            if self.min_track_length < 2:
                axis = self.time_axis if feature == 'appearance' else 1
                batch_feature_1 = np.squeeze(batch_feature_1, axis=axis)
                batch_feature_2 = np.squeeze(batch_feature_2, axis=axis)

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
                (-width_shift_range, +width_shift_range)
            With width_shift_range=2 possible values are ints [-1, 0, +1],
            same as with width_shift_range=[-1, 0, +1],
            while with width_shift_range=1.0 possible values are floats in
            the interval [-1.0, +1.0).
        shear_range: float, shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: float or [lower, upper], Range for random zoom.
            If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].
        channel_shift_range: float, range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'. Points outside the boundaries of the input
            are filled according to the given mode:
                'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
                'nearest':  aaaaaaaa|abcd|dddddddd
                'reflect':  abcddcba|abcd|dcbaabcd
                'wrap':  abcdabcd|abcd|abcdabcd
        cval: float or int, value used for points outside the boundaries
            when fill_mode = "constant".
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
                (samples, height, width, channels),
            "channels_first" mode means that the images should have shape
                (samples, channels, height, width).
            It defaults to the image_data_format value found in your
                Keras config file at "~/.keras/keras.json".
            If you never set it, then it will be "channels_last".
        validation_split: float, fraction of images reserved for validation
            (strictly between 0 and 1).
    """

    def flow(self,
             train_dict,
             compute_shapes=guess_shapes,
             min_objects=3,
             num_classes=1,
             clear_borders=False,
             include_masks=False,
             panoptic=False,
             include_mask_transforms=True,
             transforms=['watershed'],
             transforms_kwargs={},
             anchor_params=None,
             pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
             batch_size=32,
             shuffle=False,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Generates batches of augmented/normalized data with given arrays.

        Args:
            train_dict: dictionary of X and y tensors. Both should be rank 4.
            compute_shapes: function to determine the shapes of the anchors
            min_classes: images with fewer than 'min_objects' are ignored
            num_classes: number of classes to predict
            clear_borders: boolean, whether to use clear_border on y.
            include_masks: boolean, train on mask data (MaskRCNN).
            batch_size: int (default: 1).
            shuffle: boolean (default: True).
            seed: int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: str (default: ""). Prefix to use for filenames of
                saved pictures (only relevant if save_to_dir is set).
            save_format: one of "png", "jpeg". Default: "png".
                (only relevant if save_to_dir is set)

        Returns:
            An Iterator yielding tuples of (x, y) where x is a numpy array
            of image data and y is a numpy array of labels of the same shape.
        """
        return RetinaNetIterator(
            train_dict,
            self,
            compute_shapes=compute_shapes,
            min_objects=min_objects,
            num_classes=num_classes,
            clear_borders=clear_borders,
            include_masks=include_masks,
            panoptic=panoptic,
            include_mask_transforms=include_mask_transforms,
            transforms=transforms,
            transforms_kwargs=transforms_kwargs,
            anchor_params=anchor_params,
            pyramid_levels=pyramid_levels,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class RetinaNetIterator(Iterator):
    """Iterator yielding data from Numpy arrayss (X and y).

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        train_dict: dictionary consisting of numpy arrays for X and y.
        image_data_generator: Instance of ImageDataGenerator
            to use for random transformations and normalization.
        compute_shapes: functor for generating shapes, based on the model.
        min_objects: Integer, image with fewer than min_objects are ignored.
        num_classes: Integer, number of classes for classification.
        clear_borders: Boolean, whether to call clear_border on y.
        include_masks: Boolean, whether to yield mask data.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of 'channels_first', 'channels_last'.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if save_to_dir is set).
        save_format: Format to use for saving sample images
            (if save_to_dir is set).
    """

    def __init__(self,
                 train_dict,
                 image_data_generator,
                 compute_shapes=guess_shapes,
                 anchor_params=None,
                 pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                 min_objects=3,
                 num_classes=1,
                 clear_borders=False,
                 include_masks=False,
                 panoptic=False,
                 include_mask_transforms=True,
                 transforms=['watershed'],
                 transforms_kwargs={},
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
        self.anchor_params = anchor_params
        self.pyramid_levels = [int(l[1:]) for l in pyramid_levels]
        self.min_objects = min_objects
        self.num_classes = num_classes
        self.include_masks = include_masks
        self.panoptic = panoptic
        self.include_mask_transforms = include_mask_transforms
        self.transforms = transforms
        self.transforms_kwargs = transforms_kwargs
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        # Add semantic segmentation targets if panoptic segmentation
        # flag is True
        if panoptic:
            # Create a list of all the semantic targets. We need to be able
            # to have multiple semantic heads
            y_semantic_list = []
            # Add all the keys that contain y_semantic
            for key in train_dict:
                if 'y_semantic' in key:
                    y_semantic_list.append(train_dict['y_semantic'])

            if include_mask_transforms:
                # Check whether transform_kwargs_dict has an entry
                for transform in transforms:
                    if transform not in transforms_kwargs:
                        transforms_kwargs[transform] = {}

                # Add transformed masks
                for transform in transforms:
                    transform_kwargs = transforms_kwargs[transform]
                    y_transform = _transform_masks(y, transform,
                                                   data_format=data_format,
                                                   **transform_kwargs)
                    y_semantic_list.append(y_transform)

            self.y_semantic_list = [np.asarray(y_semantic, dtype='int32')
                                    for y_semantic in y_semantic_list]

        invalid_batches = []
        # Remove images with small numbers of cells
        for b in range(self.x.shape[0]):
            y_batch = np.squeeze(self.y[b], axis=self.channel_axis - 1)
            y_batch = clear_border(y_batch) if clear_borders else y_batch
            y_batch = np.expand_dims(y_batch, axis=self.channel_axis - 1)

            self.y[b] = y_batch

            if len(np.unique(self.y[b])) - 1 < self.min_objects:
                invalid_batches.append(b)

        invalid_batches = np.array(invalid_batches, dtype='int')

        if invalid_batches.size > 0:
            logging.warning('Removing %s of %s images with fewer than %s '
                            'objects.', invalid_batches.size, self.x.shape[0],
                            self.min_objects)

        self.y = np.delete(self.y, invalid_batches, axis=0)
        self.x = np.delete(self.x, invalid_batches, axis=0)

        if self.panoptic:
            self.y_semantic_list = [np.delete(y, invalid_batches, axis=0)
                                    for y in self.y_semantic_list]

        super(RetinaNetIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def filter_annotations(self, image, annotations):
        """Filter annotations by removing those that are outside of the
        image bounds or whose width/height < 0.

        Args:
            image: ndarray, the raw image data.
            annotations: dict of annotations including labels and bboxes
        """
        row_axis = 1 if self.data_format == 'channels_first' else 0
        invalid_indices = np.where(
            (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
            (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
            (annotations['bboxes'][:, 0] < 0) |
            (annotations['bboxes'][:, 1] < 0) |
            (annotations['bboxes'][:, 2] > image.shape[row_axis + 1]) |
            (annotations['bboxes'][:, 3] > image.shape[row_axis])
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
            dict: annotations of bboxes and labels
        """
        labels, bboxes, masks = [], [], []
        for prop in regionprops(np.squeeze(y.astype('int'))):
            y1, x1, y2, x2 = prop.bbox
            bboxes.append([x1, y1, x2, y2])
            labels.append(0)  # boolean object detection
            masks.append(np.where(y == prop.label, 1, 0))

        labels = np.array(labels)
        bboxes = np.array(bboxes)
        masks = np.array(masks).astype('uint8')

        # reshape bboxes in case it is empty.
        bboxes = np.reshape(bboxes, (bboxes.shape[0], 4))

        annotations = {'labels': labels, 'bboxes': bboxes}

        if self.include_masks:
            annotations['masks'] = masks

        annotations = self.filter_annotations(y, annotations)
        return annotations

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        if self.panoptic:
            batch_y_semantic_list = [np.zeros(tuple([len(index_array)] + list(ys.shape[1:])))
                                     for ys in self.y_semantic_list]

        annotations_list = []

        max_shape = []

        for i, j in enumerate(index_array):
            x = self.x[j]
            y = self.y[j]

            if self.panoptic:
                y_semantic_list = [y_semantic[j] for y_semantic in self.y_semantic_list]

            # Apply transformation
            if self.panoptic:
                x, y_list = self.image_data_generator.random_transform(x, [y] + y_semantic_list)
                y = y_list[0]
                y_semantic_list = y_list[1:]
            else:
                x, y = self.image_data_generator.random_transform(x, y)

            # Find max shape of image data.  Used for masking.
            if not max_shape:
                max_shape = list(x.shape)
            else:
                for k in range(len(x.shape)):
                    if x.shape[k] > max_shape[k]:
                        max_shape[k] = x.shape[k]

            # Get the bounding boxes from the transformed masks!
            annotations = self.load_annotations(y)
            annotations_list.append(annotations)

            x = self.image_data_generator.standardize(x)

            batch_x[i] = x

            if self.panoptic:
                for k in range(len(y_semantic_list)):
                    batch_y_semantic_list[k][i] = y_semantic_list[k]

        anchors = anchors_for_shape(
            batch_x.shape[1:],
            pyramid_levels=self.pyramid_levels,
            anchor_params=self.anchor_params,
            shapes_callback=self.compute_shapes)

        regressions, labels = anchor_targets_bbox(
            anchors,
            batch_x,
            annotations_list,
            self.num_classes)

        max_shape = tuple(max_shape)  # was a list for max shape indexing

        if self.include_masks:
            # masks_batch has shape: (batch size, max_annotations,
            #     bbox_x1 + bbox_y1 + bbox_x2 + bbox_y2 + label +
            #     width + height + max_image_dimension)
            max_annotations = max(len(a['masks']) for a in annotations_list)
            masks_batch_shape = (len(index_array), max_annotations,
                                 5 + 2 + max_shape[0] * max_shape[1])
            masks_batch = np.zeros(masks_batch_shape, dtype=K.floatx())

            for i, ann in enumerate(annotations_list):
                masks_batch[i, :ann['bboxes'].shape[0], :4] = ann['bboxes']
                masks_batch[i, :ann['labels'].shape[0], 4] = ann['labels']
                masks_batch[i, :, 5] = max_shape[1]  # width
                masks_batch[i, :, 6] = max_shape[0]  # height

                # add flattened mask
                for j, mask in enumerate(ann['masks']):
                    masks_batch[i, j, 7:] = mask.flatten()

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

        batch_outputs = [regressions, labels]
        if self.include_masks:
            batch_outputs.append(masks_batch)
        if self.panoptic:
            batch_outputs.extend(batch_y_semantic_list)

        return batch_x, batch_outputs

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


class ScaleIterator(Iterator):
    """Iterator yielding data from Numpy arrayss (`X and `y`).

    Args:
        train_dict: dictionary consisting of numpy arrays for X and y.
        image_data_generator: Instance of ImageDataGenerator
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        window_size: size of sampling window around each pixel
        balance_classes: balance class representation when sampling
        max_class_samples: maximum number of samples per class.
        seed: Random seed for data shuffling.
        data_format: String, one of 'channels_first', 'channels_last'.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if save_to_dir is set).
        save_format: Format to use for saving sample images
            (if save_to_dir is set).
    """

    def __init__(self,
                 train_dict,
                 scale_generator,
                 batch_size=1,
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
        self.x = np.asarray(X, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageFullyConvIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)

        self.y = np.ones((self.x.shape[0], 1), dtype=K.floatx())
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.scale_generator = scale_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ScaleIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:]))

        for i, j in enumerate(index_array):
            x = self.x[j]
            y = self.y[j]

            x, y = self.scale_generator.random_transform(x.astype(K.floatx()), y)

            x = self.scale_generator.standardize(x)

            batch_x[i] = x
            batch_y[i] = y

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


class ScaleDataGenerator(ImageFullyConvDataGenerator):
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
        return ScaleIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def random_transform(self, x, y=None, seed=None):
        """Applies a random transformation to an image.

        Args:
            x: 3D tensor or list of 3D tensors,
                single image.
            y: 3D tensor or list of 3D tensors,
                label mask(s) for `x`, optional.
            seed: Random seed.

        Returns:
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        params['zy'] = params['zx']

        if isinstance(x, list):
            x = [self.apply_transform(x_i, params) for x_i in x]
        else:
            x = self.apply_transform(x, params)

        if y is None:
            return x
        return x, np.array(params['zx'])
