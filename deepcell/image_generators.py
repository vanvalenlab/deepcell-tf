"""
image_generators.py

Image generators for training convolutional neural networks

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import warnings

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel_h
from skimage.filters import sobel_v
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import resize
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import random_channel_shift
from tensorflow.python.keras.preprocessing.image import apply_transform
from tensorflow.python.keras.preprocessing.image import flip_axis
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from .utils.transform_utils import transform_matrix_offset_center

"""
Custom image generators
"""

class ImageSampleArrayIterator(Iterator):
    def __init__(self, train_dict, image_data_generator,
                 batch_size=32, shuffle=False, seed=None, data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):

        if train_dict['y'].size > 0 and len(train_dict['pixels_x']) != len(train_dict['y']):
            raise Exception('Number of sampled pixels and y (labels) should have the same length. '
                            'Found: Number of sampled pixels = {}, y.shape = {}'.format(
                                len(train_dict['pixels_x']), np.asarray(train_dict['y']).shape))
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` should have rank 4. '
                             'You passed an array with shape', self.x.shape)
        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.y = train_dict['y']
        self.b = train_dict['batch']
        self.pixels_x = train_dict['pixels_x']
        self.pixels_y = train_dict['pixels_y']
        self.win_x = train_dict['win_x']
        self.win_y = train_dict['win_y']
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(ImageSampleArrayIterator, self).__init__(
            len(train_dict['y']), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.channel_axis == 1:
            batch_x = np.zeros((len(index_array), self.x.shape[self.channel_axis], 2*self.win_x + 1, 2*self.win_y + 1))
        else:
            batch_x = np.zeros((len(index_array), 2*self.win_x + 1, 2*self.win_y + 1, self.x.shape[self.channel_axis]))

        for i, j in enumerate(index_array):
            batch = self.b[j]
            pixel_x = self.pixels_x[j]
            pixel_y = self.pixels_y[j]
            win_x = self.win_x
            win_y = self.win_y

            if self.channel_axis == 1:
                x = self.x[batch, :, pixel_x-win_x:pixel_x+win_x+1, pixel_y-win_y:pixel_y+win_y+1]
            else:
                x = self.x[batch, pixel_x-win_x:pixel_x+win_x+1, pixel_y-win_y:pixel_y+win_y+1, :]

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

        def __next__(self):
            """For python 2.x.
            # Returns the next batch.
            """
            # Keeps under lock only the mechanism which advances
            # the indexing of each batch.
            with self.lock:
                index_array = next(self.index_generator)
                # The transformation of images is not under thread lock
                # so it can be done in parallel
            return self._get_batches_of_transformed_samples(index_array)

class SampleDataGenerator(ImageDataGenerator):
    def sample_flow(self, train_dict, batch_size=32, shuffle=True, seed=None,
                    save_to_dir=None, save_prefix='', save_format='png'):
        return ImageSampleArrayIterator(
            train_dict, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed, data_format=self.data_format,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

class ImageFullyConvIterator(Iterator):
    def __init__(self, train_dict, image_data_generator,
                 batch_size=1, shuffle=False, seed=None,
                 data_format=None, target_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.win_x = train_dict['win_x']
        self.win_y = train_dict['win_y']

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` should have rank 4. '
                             'You passed an array with shape {}'.format(self.x.shape))

        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.y = train_dict['y']
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.target_format = target_format
        super(ImageFullyConvIterator, self).__init__(self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        epsilon = K.epsilon() # epsilon = 1e-8
        if self.target_format == 'direction':
            y_channel_shape = 2
        elif self.target_format == 'watershed':
            if self.channel_axis == 1:
                interior = self.y[0, 1, :, :]
            else:
                interior = self.y[0, :, :, 1]
            distance = ndi.distance_transform_edt(interior)
            min_dist = np.amin(distance.flatten())
            max_dist = np.amax(distance.flatten())
            bins = np.linspace(min_dist - epsilon, max_dist + epsilon, num=16)
            distance = np.digitize(distance, bins)
            y_channel_shape = np.amax(distance.flatten()) + 1
        else:
            y_channel_shape = self.y.shape[self.channel_axis]

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

            if self.target_format == 'direction':
                if self.channel_axis == 1:
                    interior = y[1, :, :]
                else:
                    interior = y[:, :, 1]
                distance = ndi.distance_transform_edt(interior)
                gradient_x = sobel_h(distance)
                gradient_y = sobel_v(distance)
                direction_x = gradient_x / np.sqrt(gradient_x ** 2 + gradient_y ** 2 + epsilon)
                direction_y = gradient_y / np.sqrt(gradient_x ** 2 + gradient_y ** 2 + epsilon)
                direction = np.stack([direction_x, direction_y], axis=0)
                y = direction

            if self.target_format == 'watershed':
                if self.channel_axis == 1:
                    interior = y[1, :, :]
                else:
                    interior = y[:, :, 1]
                distance = ndi.distance_transform_edt(interior)
                min_dist = np.amin(distance.flatten())
                max_dist = np.amax(distance.flatten())
                bins = np.linspace(min_dist - epsilon, max_dist + epsilon, num=16)
                distance = np.digitize(distance, bins)

                # convert to one hot notation
                if self.channel_axis == 1:
                    y_shape = (np.amax(distance.flatten()) + 1, self.y.shape[2], self.y.shape[3])
                else:
                    y_shape = (self.y.shape[1], self.y.shape[2], np.amax(distance.flatten()) + 1)
                y = np.zeros(y_shape)
                for label_val in range(np.amax(distance.flatten()) + 1):
                    y[label_val, :, :] = distance == label_val

            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                # TODO: handle channel_axis?
                img_x = np.expand_dims(batch_x[i, :, :, 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.target_format == 'direction' or self.target_format == 'watershed':
                    # TODO: handle channel_axis?
                    img_y = np.expand_dims(batch_y[i, :, :, 0], -1)
                    img = array_to_img(img_y, self.data_format, scale=True)
                    fname = 'y_{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e4),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        return batch_x, batch_y

    def __next__(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
            # The transformation of images is not under thread lock
            # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

class ImageFullyConvGatherIterator(Iterator):
    def __init__(self, train_dict, image_data_generator,
                 batch_size=1, training_examples=1e5, shuffle=False, seed=None,
                 data_format=None, save_to_dir=None, save_prefix='', save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = train_dict['y']
        self.win_x = train_dict['win_x']
        self.win_y = train_dict['win_y']
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` should have rank 4. '
                             'You passed an array with shape {}'.format(self.x.shape))

        if training_examples != len(self.row_index):
            raise Exception('The number of training examples should match '
                            'the size of the training data')

        # Reorganize the batch, row, and col indices
        different_batches = list(np.arange(train_dict['X'].shape[0]))
        rows_to_sample = {}
        cols_to_sample = {}
        for batch_id in different_batches:
            rows_to_sample[batch_id] = self.row_index[self.batch_index == batch_id]
            cols_to_sample[batch_id] = self.col_index[self.batch_index == batch_id]

        #Subsample the pixel coordinates
        if self.channel_axis == 1:
            expected_label_size = (self.x.shape[0], train_dict['y'].shape[1], self.x.shape[2]-2*self.win_x, self.x.shape[3] - 2*self.win_y)
        else:
            expected_label_size = (self.x.shape[0], self.x.shape[1]-2*self.win_x, self.x.shape[2] - 2*self.win_y, train_dict['y'].shape[-1])
        if train_dict['y'] is not None and train_dict['y'].shape != expected_label_size:
            raise Exception('The expected conv-net output and label image '
                            'should have the same size. Found: '
                            'expected conv-net output shape = {}, label image shape = {}'.format(
                                expected_label_size, train_dict['y'].shape))

        super(ImageFullyConvGatherIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.channel_axis == 1:
            batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:4]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array)] + list(self.y.shape)[1:4]))
        else:
            batch_x = np.zeros(tuple([len(index_array)] + [self.x.shape[2], self.x.shape[3], self.x.shape[1]]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array)] + [self.y.shape[2], self.y.shape[3], self.y.shape[1]]))

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
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.y is None:
            return batch_x
        # if self.channel_axis == 1:
        #     batch_y = np.moveaxis(batch_y, 1, 3)
        return [batch_x, j, self.pixels_x, self.pixels_y], [batch_y]

    def __next__(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
            # The transformation of images is not under thread lock
            # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

class ImageFullyConvDataGenerator(object):
    """Generate minibatches of movie data with real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap').     ault
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'.     ault is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'.
            In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 4.
            It     aults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
            """
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: {}'.format(data_format))
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or a tuple or list of two floats. '
                             'Received arg: {}'.format(zoom_range))

    def flow(self, train_dict, batch_size=1, shuffle=True, seed=None,
            save_to_dir=None, save_prefix='', save_format='png', target_format=None):
        return ImageFullyConvIterator(
            train_dict, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format, target_format=target_format,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
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
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies `featurewise_center`, '
                              'but it has not been fit on any training data. '
                              'Fit it first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies `featurewise_std_normalization`, '
                              'but it has not been fit on any training data. '
                              'Fit it first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x, labels=None, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 4D tensor, single image.
            seed: random seed.
        # Returns
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
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
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
            y = labels #np.expand_dims(labels, axis = 0)

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

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 4.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
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
            x /= (self.std + K.epsilon())

"""
Custom siamese generators
"""

class SiameseDataGenerator(ImageDataGenerator):
    def siamese_flow(self, train_dict, crop_dim=14, min_track_length=5,
                     batch_size=32, shuffle=True, seed=None, data_format=None,
                     save_to_dir=None, save_prefix='', save_format='png'):
        return SiameseIterator(train_dict, self, crop_dim=crop_dim,
                               min_track_length=min_track_length, batch_size=batch_size,
                               shuffle=shuffle, seed=seed, data_format=data_format,
                               save_to_dir=save_to_dir, save_prefix=save_prefix,
                               save_format=save_format)

class SiameseIterator(Iterator):
    def __init__(self, train_dict, image_data_generator,
                 crop_dim=14, min_track_length=5, batch_size=32, shuffle=False,
                 seed=None, data_format=None, save_to_dir=None, save_prefix='',
                 save_format='png'):
        # Identify the channel axis so the code works regardless of what dimension
        # we are using for channels - the data in the train_dict should be channels last
        if data_format is None:
            data_format = K.image_data_format()

        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 3
            self.col_axis = 4
            self.time_axis = 2
        elif data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 2
            self.col_axis = 3
            self.time_axis = 1
        self.X = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = np.int32(train_dict['y'])
        self.crop_dim = crop_dim
        self.min_track_length = min_track_length
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.track_ids = self._get_track_ids()

        super(SiameseIterator, self).__init__(self.X.shape[0], batch_size, shuffle, seed)

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
                if y_index.size > 0: # if cell is present at all
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
        # initialize batch_x_1, batch_x_2, and batch_y
        if self.data_format == 'channels_first':
            batch_shape = (len(index_array), self.X.shape[self.channel_axis], self.crop_dim, self.crop_dim)
        else:
            batch_shape = (len(index_array), self.crop_dim, self.crop_dim, self.X.shape[self.channel_axis])

        batch_x_1 = np.zeros(batch_shape, dtype=K.floatx())
        batch_x_2 = np.zeros(batch_shape, dtype=K.floatx())
        batch_y = np.zeros((len(index_array), 2), dtype=np.int32)

        for i, j in enumerate(index_array):
            # Identify which tracks are going to be selected
            track_id = self.track_ids[j]
            batch = track_id['batch']
            label_1 = track_id['label']
            tracked_frames = track_id['frames']
            frame_1 = np.random.choice(tracked_frames) # Select a frame from the track

            X = self.X[batch]
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
                all_labels = np.delete(np.unique(y), 0) # all labels in y but 0 (background)
                acceptable_labels = np.delete(all_labels, np.where(all_labels == label_1))
                is_valid_label = False
                while not is_valid_label:
                    # get a random cell label from our acceptable list
                    label_2 = np.random.choice(acceptable_labels)

                    # count number of pixels cell occupies in each frame
                    y_true = np.sum(y == label_2, axis=(
                        self.row_axis - 1, self.col_axis - 1, self.channel_axis - 1))

                    y_index = np.where(y_true > 0)[0] # get frames where cell is present
                    is_valid_label = y_index.any() # label_2 is in a frame
                    if not is_valid_label:
                        # remove invalid label from list of acceptable labels
                        acceptable_labels = np.delete(
                            acceptable_labels, np.where(acceptable_labels == label_2))

                frame_2 = np.random.choice(y_index) # get random frame with label_2

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
            appearance_shape = (X.shape[channel_axis], len(frames), self.crop_dim, self.crop_dim)
        else:
            appearance_shape = (len(frames), self.crop_dim, self.crop_dim, X.shape[channel_axis])
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
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
            # The transformation of images is not under thread lock
            # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

"""
Custom movie generators
"""

class MovieDataGenerator(object):
    """Generate minibatches of movie data with real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channel.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap').     ault
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'.     ault is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode,
            the channels dimension (the depth) is at index 1,
            in 'channels_last' mode it is at index 4.
            It     aults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None):
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
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
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or a tuple or list of two floats. '
                             'Received arg: {}'.format(zoom_range))

    def flow(self, train_dict, batch_size=1, number_of_frames=10, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return MovieArrayIterator(
            train_dict, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format, number_of_frames=number_of_frames,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
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
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies `featurewise_center`, '
                              'but it has not been fit on any training data. '
                              'Fit it first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies `featurewise_std_normalization`, '
                              'but it has not been fit on any training data. '
                              'Fit it first by calling `.fit(numpy_data)`.')

        return x

    def random_transform(self, x, label_movie=None, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 4D tensor, single image.
            seed: random seed.
        # Returns
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
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
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

        if label_movie is not None:
            y = label_movie

            if transform_matrix is not None:
                y_new = []
                h, w = y.shape[img_row_axis], y.shape[img_col_axis]
                transform_matrix_y = transform_matrix_offset_center(transform_matrix, h, w)
                for frame in range(y.shape[img_time_axis]):
                    y_trans = apply_transform(y[frame], transform_matrix_y, img_channel_axis - 1,
                                              fill_mode='constant', cval=0)
                    y_new.append(np.rint(y_trans))
                y = np.stack(y_new, axis=0)

        if transform_matrix is not None:
            x_new = []
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix_x = transform_matrix_offset_center(transform_matrix, h, w)
            for frame in range(x.shape[img_time_axis]):
                x_new.append(apply_transform(x[frame], transform_matrix_x, img_channel_axis - 1,
                                             fill_mode=self.fill_mode, cval=self.cval))
            x = np.stack(x_new)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                if label_movie is not None:
                    y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                if label_movie is not None:
                    y = flip_axis(y, img_row_axis)

        if label_movie is not None:
            return x, y

        return x

    def fit(self, x, augment=False, rounds=1, seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 5.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 4.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
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
    """
    The movie array iterator takes in a dictionary containing the training data
    Each data set contains a data movie (X) and a label movie (y)
    The label movie is the same dimension as the channel movie with each pixel
    having its corresponding prediction
    """

    def __init__(self, train_dict, movie_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None, number_of_frames=10,
                 save_to_dir=None, save_prefix='', save_format='png'):

        if train_dict['y'] is not None and train_dict['X'].shape[0] != train_dict['y'].shape[0]:
            raise Exception('Data movie and label movie should have the same size. '
                            'Found data movie size = {} and and label movie size = {}'.format(
                                train_dict['X'].shape, train_dict['y'].shape))
        if data_format is None:
            data_format = K.image_data_format()

        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = np.asarray(train_dict['y'], dtype=K.floatx())

        if self.x.ndim != 5:
            raise ValueError('Input data in `MovieArrayIterator` should have rank 5. '
                             'You passed an array with shape {}.'.format(self.x.shape))

        if self.x.shape[self.time_axis] - number_of_frames < 0:
            raise Exception('The number of frames used in each training batch should '
                            'be less than the number of frames in the training data!')

        self.number_of_frames = number_of_frames
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(MovieArrayIterator, self).__init__(len(train_dict['y']), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        # Note to self - Make sure the exact same transformation is applied to every frame in each movie
        # Note to self - Also make sure that the exact same transformation is applied to the data movie
        # and the label movie

        # index_array = index_array[0] # index_array[0] is an integer
        if self.data_format == 'channels_first':
            batch_x = np.zeros(tuple([len(index_array), self.x.shape[1], self.number_of_frames] + list(self.x.shape)[3:]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array), self.y.shape[1], self.number_of_frames] + list(self.y.shape)[3:]))

        else:
            batch_x = np.zeros(tuple([len(index_array), self.number_of_frames] + list(self.x.shape)[2:]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array), self.number_of_frames] + list(self.y.shape)[2:]))

        for i, j in enumerate(index_array):
            if self.y is not None:
                y = self.y[j]

            # Sample along the time axis
            time_start = np.random.randint(0, high=self.x.shape[self.time_axis] - self.number_of_frames)
            time_end = time_start + self.number_of_frames
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
                    x.astype(K.floatx()), label_movie=y)
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

        if self.y is None:
            return batch_x
        # batch_y = np.rollaxis(batch_y, 1, 5)
        return batch_x, batch_y

    def __next__(self):
        """For python 2.x.
        # Returns the next batch.
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
        self.win_x = train_dict['win_x']
        self.win_y = train_dict['win_y']

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` should have rank 4. '
                             'You passed an array with shape {}'.format(self.x.shape))

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
            for l in range(1, self.num_features-1):
                if self.channel_axis == 3:
                    mask = self.y[b, :, :, l]
                else:
                    mask = self.y[b, l, :, :]
                props = regionprops(label(mask))
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
            props = regionprops(label(mask))
            bboxes = [np.array(list(prop.bbox) + list(l)) for prop in props]
            bboxes = np.concatenate(bboxes, axis=0)
        return bboxes

    def anchor_targets(image_shape,
                       annotations,
                       num_classes,
                       mask_shape=None,
                       negative_overlap=0.4,
                       positive_overlap=0.5,
                       **kwargs):
        return self.anchor_targets_bbox(image_shape, annotations, num_classes,
                                        mask_shape, negative_overlap, positive_overlap, **kwargs)

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
            batch_x = np.zeros(tuple([len(index_array), self.x.shape[2], self.x.shape[3], self.x.shape[1]]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array), self.y.shape[2], self.y.shape[3], self.y.shape[1]]))

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
        # batch_y = np.rollaxis(batch_y, 1, 4)
        return batch_x, [regressions_list, labels_list]

    def __next__(self):
        """For python 2.x.
        # Returns the next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
            # The transformation of images is not under thread lock
            # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
