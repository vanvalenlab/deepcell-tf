# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""Tests for custom image data generators
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


class TestTransformMasks(test.TestCase):

    def test_no_transform(self):
        num_classes = np.random.randint(5, size=1)[0]
        # test 2D masks
        mask = np.random.randint(num_classes, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, num_classes))

        mask = np.random.randint(num_classes, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 30, 30))

        # test 3D masks
        mask = np.random.randint(num_classes, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, num_classes))

        mask = np.random.randint(num_classes, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 10, 30, 30))

    def test_fgbg_transform(self):
        num_classes = 2  # always 2 for fg and bg
        # test 2D masks
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, num_classes))

        mask = np.random.randint(3, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 30, 30))

        # test 3D masks
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, num_classes))

        mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 10, 30, 30))

    def test_deepcell_transform(self):
        num_classes = 4
        # test 2D masks
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='deepcell', data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, num_classes))

        mask = np.random.randint(3, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='deepcell', data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 30, 30))

        # test 3D masks
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='deepcell', data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, num_classes))

        mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='deepcell', data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 10, 30, 30))

    def test_watershed_transform(self):
        distance_bins = 4
        erosion_width = 1
        # test 2D masks
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='watershed',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, distance_bins))

        distance_bins = 6
        mask = np.random.randint(3, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='watershed',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, distance_bins, 30, 30))

        # test 3D masks
        distance_bins = 5
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='watershed',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, distance_bins))

        distance_bins = 4
        mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='watershed',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, distance_bins, 10, 30, 30))

    def test_disc_transform(self):
        classes = np.random.randint(5, size=1)[0]
        # test 2D masks
        mask = np.random.randint(classes, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, classes))

        mask = np.random.randint(classes, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, classes, 30, 30))

        # test 3D masks
        mask = np.random.randint(classes, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, classes))

        mask = np.random.randint(classes, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, classes, 10, 30, 30))

    def test_bad_mask(self):
        # test bad transform
        with self.assertRaises(ValueError):
            mask = np.random.randint(3, size=(5, 30, 30, 1))
            image_generators._transform_masks(mask, transform='unknown')

        # test bad channel axis 2D
        with self.assertRaises(ValueError):
            mask = np.random.randint(3, size=(5, 30, 30, 2))
            image_generators._transform_masks(mask, transform=None)

        # test bad channel axis 3D
        with self.assertRaises(ValueError):
            mask = np.random.randint(3, size=(5, 10, 30, 30, 2))
            image_generators._transform_masks(mask, transform=None)

        # test ndim < 4
        with self.assertRaises(ValueError):
            mask = np.random.randint(3, size=(5, 30, 1))
            image_generators._transform_masks(mask, transform=None)

        # test ndim > 5
        with self.assertRaises(ValueError):
            mask = np.random.randint(3, size=(5, 10, 30, 30, 10, 1))
            image_generators._transform_masks(mask, transform=None)


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
            win_x, win_y = 2, 2
            test_batches = 3

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, img_w, img_h, 3)),
                'y': np.random.randint(2, size=(test_batches, img_w, img_h, 1))
            }
            generator.flow(train_dict, window_size=(win_x, win_y),
                           max_class_samples=10)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(2, size=(*images.shape[:-1], 1))
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    window_size=(win_x, win_y),
                    balance_classes=True,
                    max_class_samples=100,
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
            win_x, win_y = 2, 2
            test_batches = 3

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, 3, img_w, img_h)),
                'y': np.random.randint(2, size=(test_batches, 1, img_w, img_h))
            }
            generator.flow(train_dict, window_size=(win_x, win_y),
                           max_class_samples=10)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            test_shape = (images.shape[0], 1, *images.shape[2:])
            train_dict['y'] = np.random.randint(2, size=test_shape)
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    window_size=(win_x, win_y),
                    balance_classes=True,
                    max_class_samples=100,
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

        win_x, win_y = 2, 2

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
            generator.flow(train_dict, window_size=(win_x, win_y))

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((32, 10, 10, 1)),
                'y': np.random.random((25, 10, 10, 1))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 10, 10, 5))
        y = np.random.randint(2, size=(32, 10, 10, 1))
        generator.flow({'X': x, 'y': y}, window_size=(win_x, win_y))

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
        frames = 24
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
            win_x, win_y, win_z = 2, 2, 2
            test_batches = 3

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, frames, img_w, img_h, 3)),
                'y': np.random.randint(2, size=(test_batches, frames, img_w, img_h, 1))
            }
            generator.flow(train_dict, window_size=(win_x, win_y, win_z),
                           max_class_samples=10)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            assert generator.random_transform(images[0]).shape == images[0].shape
            generator.fit(images, augment=True, seed=1)
            train_dict['X'] = images
            test_shape = (images.shape[0], *images.shape[1:-1], 1)
            train_dict['y'] = np.random.randint(2, size=test_shape)
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    window_size=(win_x, win_y, win_z),
                    balance_classes=True,
                    max_class_samples=100,
                    shuffle=True):
                shape = (2 * win_z + 1, 2 * win_x + 1, 2 * win_y + 1, x.shape[-1])
                self.assertEqual(x.shape[1:], shape)
                break

    def test_sample_movie_data_generator_channels_first(self):
        frames = 24
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
            win_x, win_y, win_z = 2, 2, 2
            test_batches = 3

            # Basic test before fit
            train_dict = {
                'X': np.random.random((test_batches, 3, frames, img_w, img_h)),
                'y': np.random.randint(2, size=(test_batches, 1, frames, img_w, img_h))
            }
            generator.flow(train_dict, window_size=(win_x, win_y, win_z),
                           max_class_samples=10)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            assert generator.random_transform(images[0]).shape == images[0].shape
            generator.fit(images, augment=True, seed=1)

            train_dict['X'] = images
            test_shape = (images.shape[0], 1, *images.shape[2:])
            train_dict['y'] = np.random.randint(2, size=test_shape)
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    window_size=(win_x, win_y, win_z),
                    balance_classes=True,
                    max_class_samples=100,
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
        window_size = (2, 2, 2)

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
            generator.flow(train_dict, window_size=window_size)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((32, 30, 10, 10, 1)),
                'y': np.random.random((25, 30, 10, 10, 1))
            }
            generator.flow(train_dict, window_size=window_size)

        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 30, 10, 10, 5))
        y = np.random.randint(2, size=(32, 30, 10, 10, 1))
        generator.flow({'X': x, 'y': y}, window_size=window_size)

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
                'y': np.random.random((32, 10, 10, 1)),
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
                'y': np.random.random((32, 1, 10, 10)),
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

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((32, 10, 10, 1)),
                'y': np.random.random((25, 10, 10, 1))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((32, 10, 10, 5))
        y = np.random.random((32, 10, 10, 1))
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
                'y': np.random.random((32, 30, 10, 10, 1)),
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
                'y': np.random.random((32, 1, 30, 10, 10)),
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
        y = np.random.random((32, 30, 10, 10, 1))
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
