"""
image_generators_test.py

Tests for custom image data generators

@author: David Van Valen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
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
        im = keras.preprocessing.image.array_to_img(imarray, scale=False)
        rgb_images.append(im)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = keras.preprocessing.image.array_to_img(imarray, scale=False)
        gray_images.append(im)

    return [rgb_images, gray_images]


class TestSampleDataGenerator(test.TestCase):

    def test_sample_data_generator(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(keras.preprocessing.image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.SampleDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                # zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.,
                # brightness_range=(1, 5),
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True)

            win_x, win_y = 10, 10

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 10, 10, 3)),
                'y': np.random.random((32, 10, 10, 3)),
                'win_x': win_x,
                'win_y': win_y
            }
            generator.flow(train_dict)

            # Test save image
            # temp_dir = self.get_temp_dir()
            # generator.flow(train_dict, save_to_dir=temp_dir)

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            train_dict['y'] = np.arange(images.shape[0])
            for x, _ in generator.flow(
                    train_dict,
                    shuffle=True):
                self.assertEqual(x.shape[1:], (2 * win_x + 1, 2 * win_y + 1, x.shape[-1]))
                break

    def test_sample_data_generator_channels_first(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(keras.preprocessing.image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.SampleDataGenerator(
                data_format='channels_first')

            win_x, win_y = 10, 10

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 3, 10, 10)),
                'y': np.random.random((32, 3, 10, 10)),
                'win_x': win_x,
                'win_y': win_y
            }
            generator.flow(train_dict)

            # Fit
            generator.fit(images, augment=True)
            train_dict['X'] = images
            train_dict['y'] = np.arange(images.shape[0])
            for x, _ in generator.flow(
                    train_dict,
                    shuffle=True):
                self.assertEqual(x.shape[1:], (x.shape[1], 2 * win_x + 1, 2 * win_y + 1))
                break

    def test_sample_data_generator_invalid_data(self):
        generator = image_generators.SampleDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            # zca_whitening=True,
            data_format='channels_last')

        win_x = win_y = 5

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)
        # Test flow with invalid data
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10, 5)),
                'y': np.arange(10),
                'win_x': win_x,
                'win_y': win_y
            }
            generator.flow(train_dict)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((32, 10, 10)),
                'y': np.arange(32),
                'win_x': win_x,
                'win_y': win_y
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 10, 10, 5))
        y = np.arange(32)
        generator.flow({'X': x, 'y': y, 'win_x': win_x, 'win_y': win_y})

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
                img_list.append(keras.preprocessing.image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.ImageFullyConvDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                # zca_whitening=True,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.,
                # brightness_range=(1, 5),
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

            # Test save image
            temp_dir = self.get_temp_dir()
            generator.flow(train_dict, save_to_dir=temp_dir)

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            train_dict['y'] = np.random.random(tuple(list(images.shape)[:-1] + [1]))
            for x, _ in generator.flow(
                    train_dict,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                break

    def test_fully_conv_data_generator_channels_first(self):
        for test_images in _generate_test_images():
            img_list = []
            for im in test_images:
                img_list.append(keras.preprocessing.image.img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.ImageFullyConvDataGenerator(
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((32, 3, 10, 10)),
                'y': np.random.random((32, 3, 10, 10)),
            }
            generator.flow(train_dict)

            # Test save image
            temp_dir = self.get_temp_dir()
            generator.flow(train_dict, save_to_dir=temp_dir)

            # Fit
            generator.fit(images, augment=True)
            train_dict['X'] = images
            train_dict['y'] = np.random.random(tuple([images.shape[0], 1] +
                                                     list(images.shape)[2:]))

            for x, _ in generator.flow(
                    train_dict,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                break

    def test_sample_data_generator_invalid_data(self):
        generator = image_generators.ImageFullyConvDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            # zca_whitening=True,
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
