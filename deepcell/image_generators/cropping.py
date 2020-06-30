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

from deepcell.image_generators import SemanticDataGenerator, SemanticIterator

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None

from deepcell.image_generators import _transform_masks


class CropperIterator(SemanticIterator):
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
        crop_size (tuple): Optional parameter specifying size of crop to take from image
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

        # set output size based on cropping or not
        if crop_size is not None:
            output_size = crop_size

        else:
            output_size = self.x.shape[1:3] if self.channel_axis == 3 else self.x.shape[2:4]

        self.output_size = output_size
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
        # set output size based on output shape and # of channels
        if self.channel_axis == 3:
            x_shape = tuple([len(index_array)] + list(self.output_size) + [self.x.shape[3]])
        else:
            x_shape = tuple([len(index_array)] + [self.x.shape[1]] + list(self.output_size))

        batch_x = np.zeros(x_shape)
        batch_y = []
        for y_sem in self.y_semantic_list:
            # set output shape based on output shape and transformed label shape
            if self.channel_axis == 3:
                y_shape = tuple([len(index_array)] + list(self.output_size) + [y_sem.shape[3]])
            else:
                y_shape = tuple([len(index_array)] + [y_sem.shape[1]] + list(self.output_size))

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


class CroppingDataGenerator(SemanticDataGenerator):
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
                 interpolation_order=1,
                 crop_size=None,
                 dtype='float32'):

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
                                    dtype=dtype)

        if crop_size is not None:
            if not isinstance(crop_size, (tuple, list)):
                raise ValueError("Crop size must be a list or tuple of row/col dimensions")

        self.crop_size = crop_size

        # tensorflow does not initialize interpolation_order, so we'll do it here
        self.interpolation_order = interpolation_order

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
        return CropperIterator(
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

            img_dims = img_shape[1:] if self.channel_axis == 1 else img_shape[:2]
            if img_dims == self.crop_size:
                # don't need to crop
                pass
            elif img_dims[0] == self.crop_size[0] or img_dims[1] == self.crop_size[1]:
                raise ValueError('crop_size must be a subset of both axes or exactly '
                                 ' equal to image dims')
            elif img_dims[0] < self.crop_size[0] or img_dims[1] < self.crop_size[1]:
                raise ValueError('Crop dimensions must be smaller than image dimensions')
            else:
                row_start = np.random.randint(0, img_dims[0] - self.crop_size[0])
                col_start = np.random.randint(0, img_dims[1] - self.crop_size[1])
                crop_indices = ([row_start, row_start + self.crop_size[0]],
                                [col_start, col_start + self.crop_size[1]])

        transform_parameters['crop_indices'] = crop_indices

        return transform_parameters

    def apply_transform(self, x, transform_parameters):

        if transform_parameters['crop_indices'] is not None:
            row_indices, col_indices = transform_parameters['crop_indices']
            if self.channel_axis == 1:
                x = x[:, row_indices[0]:row_indices[1], col_indices[0]:col_indices[1]]
            else:
                x = x[row_indices[0]:row_indices[1], col_indices[0]:col_indices[1], :]

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
                print("ax shape is {}".format(ax.shape))
            else:
                ax = np.zeros(
                    tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                    dtype=self.dtype)

            for r in range(rounds):
                for i in range(x.shape[0]):
                    print("X shape is {}".format(x.shape))
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
