"""
image_generators_test.py

Tests for custom image data generators

@author: David Van Valen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.platform import test

from deepcell import image_generators


def _generate_test_images():
    img_w = img_h = 21
    rgb_images = []
    gray_images = []
    for _ in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = array_to_img(imarray, scale=False)
        rgb_images.append(im)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = array_to_img(imarray, scale=False)
        gray_images.append(im)

    return [rgb_images, gray_images]


class TestSampleDataGenerator(test.TestCase):

    def test_sample_data_generator(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.SampleDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,  # TODO: shapes not aligned when True
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            img_w, img_h = 21, 21
            win_x, win_y = 5, 5
            test_batches = 8

            pixels_x = np.random.randint(win_x, img_w - win_x, size=100)
            pixels_y = np.random.randint(win_y, img_h - win_y, size=100)
            batch = np.random.randint(test_batches, size=100)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, win_x, win_y, 3)),
                'y': np.random.randint(4, size=batch.size),
                'win_x': win_x,
                'win_y': win_y,
                'pixels_x': pixels_x,
                'pixels_y': pixels_y,
                'batch': batch
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            train_dict['y'] = np.arange(images.shape[0])
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], (2 * win_x + 1, 2 * win_y + 1, x.shape[-1]))
                break

    def test_sample_data_generator_channels_first(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, -1, 1)
            generator = image_generators.SampleDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,  # TODO: shapes not aligned when True
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: converts to channels_last?
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            img_w, img_h = 21, 21
            win_x, win_y = 5, 5
            test_batches = 8

            pixels_x = np.random.randint(win_x, img_w - win_x, size=100)
            pixels_y = np.random.randint(win_y, img_h - win_y, size=100)
            batch = np.random.randint(test_batches, size=100)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, 3, img_w, img_h)),
                'y': np.random.randint(4, size=batch.size),
                'win_x': win_x,
                'win_y': win_y,
                'pixels_x': pixels_x,
                'pixels_y': pixels_y,
                'batch': batch
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True)
            train_dict['X'] = images
            train_dict['y'] = np.arange(pixels_x.size)
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], (x.shape[1], 2 * win_x + 1, 2 * win_y + 1))
                break

    def test_sample_data_generator_invalid_data(self):
        generator = image_generators.SampleDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        win_x = win_y = 5
        img_w, img_h = 21, 21
        win_x, win_y = 10, 10
        test_batches = 8

        pixels_x = np.random.randint(img_w, size=100)
        pixels_y = np.random.randint(img_h, size=100)
        batch = np.random.randint(test_batches, size=100)

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)
        # Test flow with invalid data
        # X and y size mismatch is expected, as X is image data and
        # y is same length as batch, pixels_x, pixels_y
        # with self.assertRaises(ValueError):
        #     train_dict = {
        #         'X': np.random.random((test_batches, imw_w, img_h, 5)),
        #         'y': np.arange(batch.size),
        #         'win_x': win_x,
        #         'win_y': win_y,
        #         'pixels_x': pixels_x,
        #         'pixels_y': pixels_y,
        #         'batch': batch
        #     }
        #     generator.flow(train_dict)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((test_batches, 10, 10)),
                'y': np.arange(test_batches),
                'win_x': win_x,
                'win_y': win_y,
                'pixels_x': pixels_x,
                'pixels_y': pixels_y,
                'batch': batch
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((test_batches, img_w, img_h, 5))
        y = np.arange(test_batches)
        train_dict = {
            'X': x,
            'y': y,
            'win_x': win_x,
            'win_y': win_y,
            'pixels_x': pixels_x,
            'pixels_y': pixels_y,
            'batch': batch
        }
        generator.flow(train_dict)

        with self.assertRaises(ValueError):
            generator = image_generators.SampleDataGenerator(
                data_format='unknown')

        generator = image_generators.SampleDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.SampleDataGenerator(
                zoom_range=(2, 2, 2))


class TestSampleMovieDataGenerator(test.TestCase):

    def test_sample_movie_data_generator(self):
        frames = 10
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, (batch_count, frames, *images.shape[1:]))
            generator = image_generators.SampleMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            img_w, img_h = 21, 21
            win_x, win_y, win_z = 5, 5, frames // 2
            test_batches = 8

            pixels_x = np.random.randint(win_x, img_w - win_x, size=100)
            pixels_y = np.random.randint(win_y, img_h - win_y, size=100)
            pixels_z = np.random.randint(0, frames - win_z, size=100)
            batch = np.random.randint(test_batches, size=100)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, frames, img_w, img_h, 3)),
                'y': np.random.random((test_batches, frames, img_w, img_h, 3)),
                'win_x': win_x,
                'win_y': win_y,
                'win_z': win_z,
                'pixels_x': pixels_x,
                'pixels_y': pixels_y,
                'pixels_z': pixels_z,
                'batch': batch
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            assert generator.random_transform(images[0]).shape == images[0].shape
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            train_dict['y'] = np.arange(images.shape[0])
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                shape = (2 * win_z + 1, 2 * win_x + 1, 2 * win_y + 1, x.shape[-1])
                self.assertEqual(x.shape[1:], shape)
                break

    def test_sample_movie_data_generator_channels_first(self):
        frames = 10
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, (batch_count, frames, *images.shape[1:]))
            images = np.rollaxis(images, -1, 1)
            generator = image_generators.SampleMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            img_w, img_h = 21, 21
            win_x, win_y, win_z = 5, 5, frames // 2
            test_batches = 8

            pixels_x = np.random.randint(win_x, img_w - win_x, size=100)
            pixels_y = np.random.randint(win_y, img_h - win_y, size=100)
            pixels_z = np.random.randint(0, frames - win_z, size=100)
            batch = np.random.randint(test_batches, size=100)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, 3, frames, img_w, img_h)),
                'y': np.random.random((test_batches, 3, frames, img_w, img_h)),
                'win_x': win_x,
                'win_y': win_y,
                'win_z': win_z,
                'pixels_x': pixels_x,
                'pixels_y': pixels_y,
                'pixels_z': pixels_z,
                'batch': batch
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            assert generator.random_transform(images[0]).shape == images[0].shape
            generator.fit(images, augment=True, seed=1)

            train_dict['X'] = images
            train_dict['y'] = np.arange(images.shape[0])
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                shape = (x.shape[1], 2 * win_z + 1, 2 * win_x + 1, 2 * win_y + 1)
                self.assertEqual(x.shape[1:], shape)
                break

    def test_sample_movie_data_generator_invalid_data(self):
        generator = image_generators.SampleMovieDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        win_x = win_y = win_z = 5
        img_w, img_h, frames = 10, 10, 5
        test_batches = 8

        pixels_x = np.random.randint(img_w, size=100)
        pixels_y = np.random.randint(img_h, size=100)
        pixels_z = np.random.randint(frames, size=100)
        batch = np.random.randint(test_batches, size=100)

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, img_w, img_h))
            generator.fit(x)
        # Test flow with invalid data
        # X and y size mismatch is expected, as X is image data and
        # y is same length as batch, pixels_x, pixels_y, pixels_z
        # with self.assertRaises(ValueError):
        #     train_dict = {
        #         'X': np.random.random((test_batches, frames, img_w, img_h, 5)),
        #         'y': np.arange(batch.size),
        #         'win_x': win_x,
        #         'win_y': win_y,
        #         'win_z': win_z,
        #         'pixels_x': pixels_x,
        #         'pixels_y': pixels_y,
        #         'pixels_z': pixels_z,
        #         'batch': batch
        #     }
        #     generator.flow(train_dict)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10)),
                'y': np.arange(batch.size),
                'win_x': win_x,
                'win_y': win_y,
                'win_z': win_z,
                'pixels_x': pixels_x,
                'pixels_y': pixels_y,
                'pixels_z': pixels_z,
                'batch': batch
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((test_batches, frames, img_w, img_h, 5))
        y = np.arange(batch.size)
        train_dict = {
            'X': x,
            'y': y,
            'win_x': win_x,
            'win_y': win_y,
            'win_z': win_z,
            'pixels_x': pixels_x,
            'pixels_y': pixels_y,
            'pixels_z': pixels_z,
            'batch': batch
        }
        generator.flow(train_dict)

        with self.assertRaises(ValueError):
            generator = image_generators.SampleDataGenerator(
                data_format='unknown')

        generator = image_generators.SampleDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.SampleDataGenerator(
                zoom_range=(2, 2, 2))


class TestFullyConvDataGenerator(test.TestCase):

    def test_fully_conv_data_generator(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.ImageFullyConvDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 10, 10, 3)),
                'y': np.random.random((32, 10, 10, 3)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.random(y_shape)
            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], y_shape[1:])
                break

    def test_fully_conv_data_generator_channels_first(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.ImageFullyConvDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 3, 10, 10)),
                'y': np.random.random((32, 3, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.random(y_shape)

            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], y_shape[1:])
                break

    def test_fully_conv_data_generator_invalid_data(self):
        generator = image_generators.ImageFullyConvDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10)),
                'y': np.random.random((32, 10, 10))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 10, 10, 5))
        y = np.random.random((32, 10, 10, 5))
        generator.flow({'X': x, 'y': y})

        with self.assertRaises(ValueError):
            generator = image_generators.ImageFullyConvDataGenerator(
                data_format='unknown')

        generator = image_generators.ImageFullyConvDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.ImageFullyConvDataGenerator(
                zoom_range=(2, 2, 2))

    def test_fully_conv_data_generator_fit(self):
        generator = image_generators.ImageFullyConvDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')
        # Test grayscale
        x = np.random.random((32, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 10, 10, 3))
        generator.fit(x)
        generator = image_generators.ImageFullyConvDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_first')
        # Test grayscale
        x = np.random.random((32, 1, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 10, 10))
        generator.fit(x)

    def test_batch_standardize(self):
        # ImageFullyConvDataGenerator.standardize should work on batches
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.ImageFullyConvDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                rescale=.95,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)
            generator.fit(images, augment=True)

            transformed = np.copy(images)
            for i, im in enumerate(transformed):
                transformed[i] = generator.random_transform(im, seed=1)
            transformed = generator.standardize(transformed)


class TestMovieDataGenerator(test.TestCase):

    def test_movie_data_generator(self):
        frames = 30
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, (batch_count, frames, *images.shape[1:]))
            generator = image_generators.MovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 30, 10, 10, 3)),
                'y': np.random.random((32, 30, 10, 10, 3)),
            }
            generator.flow(train_dict, frames_per_batch=1)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.random(y_shape)
            frames_per_batch = 10
            for x, y in generator.flow(
                    train_dict,
                    shuffle=True,
                    save_to_dir=temp_dir,
                    frames_per_batch=frames_per_batch):
                batch_x_shape = (frames_per_batch, *images.shape[2:])
                batch_y_shape = (frames_per_batch, *y_shape[2:])
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(y.shape[1:], batch_y_shape)
                break

    def test_movie_data_generator_channels_first(self):
        frames = 30
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, (batch_count, frames, *images.shape[1:]))
            images = np.rollaxis(images, 4, 1)
            generator = image_generators.MovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 3, 30, 10, 10)),
                'y': np.random.random((32, 3, 30, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.random(y_shape)

            frames_per_batch = 10
            for x, y in generator.flow(
                    train_dict,
                    shuffle=True,
                    save_to_dir=temp_dir,
                    frames_per_batch=frames_per_batch):
                batch_x_shape = (images.shape[1], frames_per_batch, *images.shape[3:])
                batch_y_shape = (y_shape[1], frames_per_batch, *y_shape[3:])
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(y.shape[1:], batch_y_shape)
                break

    def test_movie_data_generator_invalid_data(self):
        generator = image_generators.MovieDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10, 1)),
                'y': np.random.random((32, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((32, 30, 10, 10, 1)),
                'y': np.random.random((25, 30, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with bigger frames_per_batch than frames
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((32, 30, 10, 10, 1)),
                'y': np.random.random((32, 30, 10, 10, 1))
            }
            generator.flow(train_dict, frames_per_batch=31)

        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 30, 10, 10, 5))
        y = np.random.random((32, 30, 10, 10, 5))
        generator.flow({'X': x, 'y': y})

        with self.assertRaises(ValueError):
            generator = image_generators.MovieDataGenerator(
                data_format='unknown')

        generator = image_generators.MovieDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.MovieDataGenerator(
                zoom_range=(2, 2, 2))

    def test_movie_data_generator_fit(self):
        generator = image_generators.MovieDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')
        # Test grayscale
        x = np.random.random((32, 30, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 30, 10, 10, 3))
        generator.fit(x)
        generator = image_generators.MovieDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_first')
        # Test grayscale
        x = np.random.random((32, 1, 30, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((32, 3, 30, 10, 10))
        generator.fit(x)

    def test_batch_standardize(self):
        # MovieDataGenerator.standardize should work on batches
        frames = 30
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, (batch_count, frames, *images.shape[1:]))
            generator = image_generators.MovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)
            generator.fit(images, augment=True)

            transformed = np.copy(images)
            for i, im in enumerate(transformed):
                transformed[i] = generator.random_transform(im, seed=1)
            transformed = generator.standardize(transformed)


class TestWatershedDataGenerator(test.TestCase):

    def test_watershed_data_generator(self):
        distance_bins = 4
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.WatershedDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 10, 10, 3)),
                'y': np.random.random((32, 10, 10, 3)),
            }
            generator.flow(train_dict, distance_bins=distance_bins)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            train_dict['y'] = np.random.random(tuple(list(images.shape)[:-1] + [3]))
            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    distance_bins=distance_bins,
                    shuffle=True):
                shape = tuple(list(images.shape)[1:-1] + [distance_bins])
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], shape)
                break

    def test_watershed_data_generator_channels_first(self):
        distance_bins = 4
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.WatershedDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 3, 10, 10)),
                'y': np.random.random((32, 3, 10, 10)),
            }
            generator.flow(train_dict, distance_bins=distance_bins)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            # y expects 3 channels, cell edge, cell interior, background
            train_dict['y'] = np.random.random(tuple([images.shape[0], 3] +
                                                     list(images.shape)[2:]))

            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    distance_bins=distance_bins,
                    shuffle=True):
                shape = tuple([distance_bins] + list(images.shape)[2:])
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], shape)
                break

    def test_watershed_data_generator_invalid_data(self):
        distance_bins = 4
        generator = image_generators.WatershedDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10)),
                'y': np.random.random((32, 10, 10))
            }
            generator.flow(train_dict, distance_bins=distance_bins)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 10, 10, 5))
        y = np.random.random((32, 10, 10, 5))
        generator.flow({'X': x, 'y': y}, distance_bins=distance_bins)

        with self.assertRaises(ValueError):
            generator = image_generators.WatershedDataGenerator(
                data_format='unknown')

        generator = image_generators.WatershedDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.WatershedDataGenerator(
                zoom_range=(2, 2, 2))


class TestWatershedMovieDataGenerator(test.TestCase):

    def test_watershed_movie_data_generator(self):
        frames = 5
        distance_bins = 4
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, (batch_count, frames, *images.shape[1:]))
            generator = image_generators.WatershedMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, frames, 10, 10, 3)),
                'y': np.random.random((32, frames, 10, 10, 3)),
            }
            generator.flow(train_dict, frames_per_batch=1, distance_bins=distance_bins)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(2, size=y_shape)
            frames_per_batch = 2
            for x, y in generator.flow(
                    train_dict,
                    shuffle=True,
                    save_to_dir=temp_dir,
                    distance_bins=distance_bins,
                    frames_per_batch=frames_per_batch):
                batch_x_shape = (frames_per_batch, *images.shape[2:])
                batch_y_shape = (frames_per_batch, *y_shape[2:-1], distance_bins)
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(y.shape[1:], batch_y_shape)
                break

    def test_watershed_movie_data_generator_channels_first(self):
        frames = 5
        distance_bins = 4
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, (batch_count, frames, *images.shape[1:]))
            images = np.rollaxis(images, 4, 1)
            generator = image_generators.WatershedMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 3, frames, 10, 10)),
                'y': np.random.random((32, 3, frames, 10, 10)),
            }
            generator.flow(train_dict, frames_per_batch=1, distance_bins=distance_bins)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(2, size=y_shape)
            frames_per_batch = 2
            for x, y in generator.flow(
                    train_dict,
                    shuffle=True,
                    save_to_dir=temp_dir,
                    distance_bins=distance_bins,
                    frames_per_batch=frames_per_batch):
                batch_x_shape = (images.shape[1], frames_per_batch, *images.shape[3:])
                batch_y_shape = (distance_bins, frames_per_batch, *y_shape[3:])
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(y.shape[1:], batch_y_shape)
                break

    def test_watershed_movie_data_generator_invalid_data(self):
        distance_bins = 4
        generator = image_generators.WatershedMovieDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10, 1)),
                'y': np.random.random((32, 10, 10, 1))
            }
            generator.flow(train_dict, distance_bins=distance_bins)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((32, 30, 10, 10, 1)),
                'y': np.random.random((25, 30, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with bigger frames_per_batch than frames
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((32, 30, 10, 10, 1)),
                'y': np.random.random((32, 30, 10, 10, 1))
            }
            generator.flow(train_dict, frames_per_batch=31, distance_bins=distance_bins)

        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 5, 10, 10, 5))
        y = np.random.random((32, 5, 10, 10, 5))
        generator.flow({'X': x, 'y': y}, frames_per_batch=2)

        with self.assertRaises(ValueError):
            generator = image_generators.WatershedMovieDataGenerator(
                data_format='unknown')

        generator = image_generators.WatershedMovieDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.WatershedMovieDataGenerator(
                zoom_range=(2, 2, 2))


class TestDiscDataGenerator(test.TestCase):

    def test_disc_data_generator(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.DiscDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 10, 10, 3)),
                'y': np.random.random((32, 10, 10, 3)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            # y expects 3 channels, cell edge, cell interior, background
            train_dict['y'] = np.random.random(tuple(list(images.shape)[:-1] + [3]))

            # TODO: Why does this work?
            y_channel = np.ceil(np.amax(train_dict['y'][:, 1, :, :]))

            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                shape = tuple(list(images.shape)[1:-1] + [y_channel])
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], shape)
                break

    def test_disc_data_generator_channels_first(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.DiscDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 3, 10, 10)),
                'y': np.random.random((32, 3, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            # y expects 3 channels, cell edge, cell interior, background
            train_dict['y'] = np.random.random(tuple([images.shape[0], 3] +
                                                     list(images.shape)[2:]))

            # TODO: Why does this work?
            y_channel = np.ceil(np.amax(train_dict['y'][:, 1, :, :]))

            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                shape = tuple([y_channel] + list(images.shape)[2:])
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], shape)
                break

    def test_disc_data_generator_invalid_data(self):
        generator = image_generators.DiscDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10)),
                'y': np.random.random((32, 10, 10))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 10, 10, 5))
        y = np.random.random((32, 10, 10, 5))
        generator.flow({'X': x, 'y': y})

        with self.assertRaises(ValueError):
            generator = image_generators.DiscDataGenerator(
                data_format='unknown')

        generator = image_generators.DiscDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.DiscDataGenerator(
                zoom_range=(2, 2, 2))
