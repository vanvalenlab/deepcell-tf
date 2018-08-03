"""
image_generators.py

Image generators for training convolutional neural networks

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from fnmatch import fnmatch

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel_h
from skimage.filters import sobel_v
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.io import imread

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import random_channel_shift
from tensorflow.python.keras.preprocessing.image import apply_transform
from tensorflow.python.keras.preprocessing.image import flip_axis
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from keras_retinanet.preprocessing.generator import Generator as _RetinaNetGenerator
from keras_maskrcnn.preprocessing.generator import Generator as _MaskRCNNGenerator

from .utils.transform_utils import transform_matrix_offset_center
from .utils.transform_utils import distance_transform_2d
from .utils.retinanet_anchor_utils import anchor_targets_bbox


"""
Custom image generators
"""


class ImageSampleArrayIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if train_dict['X'].shape[0] != train_dict['y'].shape[0]:
            raise ValueError('Training batches and labels should have the same'
                             'length. Found X.shape: {} y.shape: {}'.format(
                                 train_dict['X'].shape, train_dict['y'].shape))
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageSampleArrayIterator` should'
                             'have rank 4. Got array with shape', self.x.shape)

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.y = train_dict['y']
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
            x = self.x[j]
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
    def flow(self,
             train_dict,
             batch_size=32,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return ImageSampleArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class ImageFullyConvIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 target_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `ImageFullyConvIterator` should'
                             'have rank 4. Got array with shape', self.x.shape)

        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.y = np.array(train_dict['y'], dtype='int32')
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.target_format = target_format
        super(ImageFullyConvIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        epsilon = K.epsilon()
        if self.target_format == 'direction':
            y_channel_shape = 2
        else:
            y_channel_shape = self.y.shape[self.channel_axis]

        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        if self.channel_axis == 1:
            batch_y = np.zeros(tuple([len(index_array), y_channel_shape] +
                                     list(self.y.shape)[2:]))
        else:
            batch_y = np.zeros(tuple([len(index_array)] +
                                     list(self.y.shape)[1:3] +
                                     [y_channel_shape]))

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
                norm = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + epsilon)
                direction_x = gradient_x / norm
                direction_y = gradient_y / norm
                direction = np.stack([direction_x, direction_y], axis=0)
                y = direction

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

                if self.target_format == 'direction':
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


class ImageFullyConvDataGenerator(ImageDataGenerator):
    """
    Generate minibatches of image data and masks
    with real-time data augmentation.
    """

    def flow(self,
             train_dict,
             batch_size=1,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             target_format=None):
        return ImageFullyConvIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            target_format=target_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

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
        batch_y = np.zeros((len(index_array), 2), dtype=np.int32)

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
Watershed generator
"""


class WatershedDataGenerator(ImageFullyConvDataGenerator):
    def flow(self,
             train_dict,
             batch_size=1,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             distance_bins=16,
             save_format='png'):
        return WatershedIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            distance_bins=distance_bins,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class WatershedIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 distance_bins=16,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError(
                'Input data in `WatershedIterator` should have rank 4. '
                'You passed an array with shape', self.x.shape)

        self.channel_axis = 3 if data_format == 'channels_last' else 1
        self.distance_bins = distance_bins
        self.y = train_dict['y']
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(WatershedIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        if self.channel_axis == 1:
            batch_y = np.zeros(tuple([len(index_array), self.distance_bins] +
                                     list(self.y.shape)[2:]))
        else:
            batch_y = np.zeros(tuple([len(index_array)] +
                                     list(self.y.shape)[1:3] +
                                     [self.distance_bins]))

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            if self.channel_axis == 1:
                interior = y[1, :, :]
            else:
                interior = y[:, :, 1]

            distance = distance_transform_2d(interior, self.distance_bins)

            # convert to one hot notation
            if self.channel_axis == 1:
                y_shape = (self.distance_bins, self.y.shape[2], self.y.shape[3])
            else:
                y_shape = (self.y.shape[1], self.y.shape[2], self.distance_bins)

            y = np.zeros(y_shape)
            for label_val in range(np.amax(distance) + 1):
                if self.channel_axis == 1:
                    y[label_val, :, :] = distance == label_val
                else:
                    y[:, :, label_val] = distance == label_val

            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                # Save X batch
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.y is not None:
                    # Save y batch, but just the MAX distance for each pixel
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
        return batch_x, batch_y

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


class MovieDataGenerator(ImageDataGenerator):
    """Generate minibatches of movie data with real-time data augmentation."""

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
             number_of_frames=10,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return MovieArrayIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            number_of_frames=number_of_frames,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

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

    def random_transform(self, x, label_movie=None, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 5D tensor, single image.
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

    def __init__(self,
                 train_dict,
                 movie_data_generator,
                 batch_size=32,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 number_of_frames=10,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):

        if train_dict['y'] is not None and train_dict['X'].shape[0] != train_dict['y'].shape[0]:
            raise ValueError(
                'Data movie and label movie should have the same size. '
                'Found data movie size = {} and and label movie size = {}'.format(
                    train_dict['X'].shape, train_dict['y'].shape))

        if data_format is None:
            data_format = K.image_data_format()

        self.channel_axis = 4 if data_format == 'channels_last' else 1
        self.time_axis = 1 if data_format == 'channels_last' else 2
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())
        self.y = np.asarray(train_dict['y'], dtype=K.floatx())

        if self.x.ndim != 5:
            raise ValueError(
                'Input data in `MovieArrayIterator` should have rank 5. '
                'You passed an array with shape', self.x.shape)

        if self.x.shape[self.time_axis] - number_of_frames < 0:
            raise ValueError(
                'The number of frames used in each training batch should '
                'be less than the number of frames in the training data!')

        self.number_of_frames = number_of_frames
        self.movie_data_generator = movie_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(MovieArrayIterator, self).__init__(
            len(train_dict['y']), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format == 'channels_first':
            batch_x = np.zeros((len(index_array),
                                self.x.shape[1],
                                self.number_of_frames,
                                self.x.shape[3],
                                self.x.shape[4]))
            if self.y is not None:
                batch_y = np.zeros((len(index_array),
                                    self.y.shape[1],
                                    self.number_of_frames,
                                    self.y.shape[3],
                                    self.y.shape[3]))

        else:
            batch_x = np.zeros(tuple([len(index_array), self.number_of_frames] +
                                     list(self.x.shape)[2:]))
            if self.y is not None:
                batch_y = np.zeros(tuple([len(index_array), self.number_of_frames] +
                                         list(self.y.shape)[2:]))

        for i, j in enumerate(index_array):
            if self.y is not None:
                y = self.y[j]

            # Sample along the time axis
            last_frame = self.x.shape[self.time_axis] - self.number_of_frames
            time_start = np.random.randint(0, high=last_frame)
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
        return batch_x, batch_y

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
            raise ValueError(
                'Input data in `BoundingBoxIterator` should have rank 4. '
                'You passed an array with shape', self.x.shape)

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
Discriminative generator
"""


class DiscDataGenerator(ImageFullyConvDataGenerator):

    def flow(self,
             train_dict,
             batch_size=1,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        return DiscIterator(
            train_dict,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class DiscIterator(Iterator):
    def __init__(self,
                 train_dict,
                 image_data_generator,
                 batch_size=1,
                 shuffle=False,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):
        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(train_dict['X'], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError(
                'Input data in `DiscIterator` should have rank 4. '
                'You passed an array with shape', self.x.shape)

        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.y = train_dict['y']
        self.max_label = 0
        for batch in range(self.y.shape[0]):
            if self.channel_axis == 1:
                label_matrix = label(self.y[batch, 1, :, :])
            else:
                label_matrix = label(self.y[batch, :, :, 1])
            max_label = np.amax(label_matrix)
            if max_label > self.max_label:
                self.max_label = max_label

        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(DiscIterator, self).__init__(
            self.x.shape[0], batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.x.shape)[1:]))
        if self.channel_axis == 1:
            batch_y = np.zeros(tuple([len(index_array), self.max_label + 1] +
                                     list(self.y.shape)[2:]))
        else:
            batch_y = np.zeros(tuple([len(index_array)] +
                                     list(self.y.shape)[1:3] +
                                     [self.max_label + 1]))

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.y is not None:
                y = self.y[j]
                x, y = self.image_data_generator.random_transform(x.astype(K.floatx()), y)
            else:
                x = self.image_data_generator.random_transform(x.astype(K.floatx()))

            x = self.image_data_generator.standardize(x)

            if self.channel_axis == 1:
                interior = y[1, :, :]
            else:
                interior = y[:, :, 1]

            label_matrix = label(interior)

            # convert to one hot notation
            if self.channel_axis == 1:
                y_shape = (self.max_label + 1, self.y.shape[2], self.y.shape[3])
            else:
                y_shape = (self.y.shape[1], self.y.shape[2], self.max_label + 1)

            y = np.zeros(y_shape)

            for label_val in range(self.max_label + 1):
                if self.channel_axis == 1:
                    y[label_val, :, :] = label_matrix == label_val
                else:
                    y[:, :, label_val] = label_matrix == label_val

            batch_x[i] = x
            batch_y[i] = y

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                # Save X batch
                img_x = np.expand_dims(batch_x[i, :, :, 0], -1)
                img = array_to_img(img_x, self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

                if self.y is not None:
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
