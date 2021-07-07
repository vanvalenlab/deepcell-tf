# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
"""Tests for custom image data generators"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage as sk

from PIL import Image

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.platform import test

from deepcell import image_generators


def all_test_images():
    img_w = img_h = 20
    rgb_images = []
    rgba_images = []
    gray_images = []
    for n in range(8):
        bias = np.random.rand(img_w, img_h, 1) * 64
        variance = np.random.rand(img_w, img_h, 1) * (255 - 64)
        imarray = np.random.rand(img_w, img_h, 3) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        rgb_images.append(im)

        imarray = np.random.rand(img_w, img_h, 4) * variance + bias
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        rgba_images.append(im)

        imarray = np.random.rand(img_w, img_h, 1) * variance + bias
        im = Image.fromarray(
            imarray.astype('uint8').squeeze()).convert('L')
        gray_images.append(im)

    return [rgb_images, rgba_images, gray_images]


class TestTransformMasks(test.TestCase):

    def test_no_transform(self):
        num_classes = np.random.randint(5, size=1)[0]
        # test 2D masks
        mask = np.random.randint(num_classes, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, num_classes))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        mask = np.random.randint(num_classes, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        # test 3D masks
        mask = np.random.randint(num_classes, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, num_classes))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        mask = np.random.randint(num_classes, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform=None, data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 10, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

    def test_fgbg_transform(self):
        num_classes = 2  # always 2 for fg and bg
        # test 2D masks
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, num_classes))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        mask = np.random.randint(3, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        # test 3D masks
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, num_classes))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='fgbg', data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, num_classes, 10, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

    def test_pixelwise_transform(self):
        num_classes = 3
        # test 2D masks
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='pixelwise', data_format='channels_last',
            separate_edge_classes=True)
        self.assertEqual(mask_transform.shape, (5, 30, 30, 4))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        mask = np.random.randint(3, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='pixelwise', data_format='channels_first',
            separate_edge_classes=False)
        self.assertEqual(mask_transform.shape, (5, 3, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        # test 3D masks
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask, transform='pixelwise', data_format='channels_last',
            separate_edge_classes=False)
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, 3))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask, transform='pixelwise', data_format='channels_first',
            separate_edge_classes=True)
        self.assertEqual(mask_transform.shape, (5, 4, 10, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

    def test_outer_distance_transform(self):
        K.set_floatx('float16')
        # test 2D masks
        distance_bins = None
        erosion_width = 1
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='outer-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, 1))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

        distance_bins = 4
        erosion_width = 1
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='outer-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, distance_bins))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        distance_bins = 6
        mask = np.random.randint(3, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='outer-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, distance_bins, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        # test 3D masks
        K.set_floatx('float32')
        distance_bins = None
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='outer-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, 1))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

        distance_bins = 5
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='outer-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, distance_bins))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        distance_bins = 4
        mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='outer-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, distance_bins, 10, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

    def test_inner_distance_transform(self):
        K.set_floatx('float16')

        # test 2D masks
        distance_bins = None
        erosion_width = 1
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='inner-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, 1))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

        distance_bins = 4
        erosion_width = 1
        mask = np.random.randint(3, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='inner-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, distance_bins))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        distance_bins = 6
        mask = np.random.randint(3, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='inner-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, distance_bins, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        # test 3D masks
        K.set_floatx('float32')
        distance_bins = None
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='inner-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, 1))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

        distance_bins = 5
        mask = np.random.randint(3, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='inner-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, distance_bins))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

        distance_bins = 4
        mask = np.random.randint(3, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='inner-distance',
            distance_bins=distance_bins,
            erosion_width=erosion_width,
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, distance_bins, 10, 30, 30))
        self.assertTrue(np.issubdtype(mask_transform.dtype, np.integer))

    def test_disc_transform(self):
        classes = np.random.randint(5, size=1)[0]
        # test 2D masks
        mask = np.random.randint(classes, size=(5, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 30, 30, classes))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

        mask = np.random.randint(classes, size=(5, 1, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, classes, 30, 30))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

        # test 3D masks
        mask = np.random.randint(classes, size=(5, 10, 30, 30, 1))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_last')
        self.assertEqual(mask_transform.shape, (5, 10, 30, 30, classes))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

        mask = np.random.randint(classes, size=(5, 1, 10, 30, 30))
        mask_transform = image_generators._transform_masks(
            mask,
            transform='disc',
            data_format='channels_first')
        self.assertEqual(mask_transform.shape, (5, classes, 10, 30, 30))
        self.assertEqual(mask_transform.dtype, np.dtype(K.floatx()))

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
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.SampleDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,  # dim 1 not aligned (1, 75) & (1323, 1323)
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
            test_shape = tuple(list(images.shape[:-1]) + [1])
            train_dict['y'] = np.random.randint(2, size=test_shape)
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
        for test_images in all_test_images():
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
                zca_whitening=False,  # dim 0 not aligned (1, 75) & (1323, 1323)
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
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
            test_shape = tuple([images.shape[0], 1] + list(images.shape[2:]))
            train_dict['y'] = np.random.randint(2, size=test_shape)
            for x, _ in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    window_size=(win_x, win_y),
                    balance_classes=True,
                    max_class_samples=10,
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
                'X': np.random.random((8, 10, 10)),
                'y': np.random.random((8, 10, 10))
            }
            generator.flow(train_dict, window_size=(win_x, win_y))

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((7, 10, 10, 1))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        x = np.random.random((8, 10, 10, 5))
        y = np.random.randint(2, size=(8, 10, 10, 1))
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
        frames = 7
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, tuple([batch_count, frames] +
                                              list(images.shape[1:])))
            generator = image_generators.SampleMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
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
            test_shape = tuple([images.shape[0]] + list(images.shape[1:-1]) + [1])
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
        frames = 7
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, tuple([batch_count, frames] +
                                              list(images.shape[1:])))
            images = np.rollaxis(images, -1, 1)
            generator = image_generators.SampleMovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
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
            test_shape = tuple([images.shape[0], 1] + list(images.shape[2:]))
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
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((8, 10, 10, 1))
            }
            generator.flow(train_dict, window_size=window_size)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((7, 11, 10, 10, 1))
            }
            generator.flow(train_dict, window_size=window_size)

        # Invalid number of channels: will work but raise a warning
        x = np.random.random((8, 11, 10, 10, 5))
        y = np.random.randint(2, size=(8, 11, 10, 10, 1))
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
        for test_images in all_test_images():
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
                'X': np.random.random((8, 10, 10, 3)),
                'y': np.random.random((8, 10, 10, 1)),
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
                    skip=1,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(len(y), 2)
                self.assertEqual(y[0].shape[1:], y_shape[1:])
                self.assertEqual(y[-1].shape[1:], y_shape[1:])
                break

    def test_fully_conv_data_generator_channels_first(self):
        for test_images in all_test_images():
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 10, 10)),
                'y': np.random.random((8, 1, 10, 10)),
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
                    skip=1,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(len(y), 2)
                self.assertEqual(y[0].shape[1:], y_shape[1:])
                self.assertEqual(y[-1].shape[1:], y_shape[1:])
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
                'X': np.random.random((8, 10, 10)),
                'y': np.random.random((8, 10, 10))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((7, 10, 10, 1))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 10, 10, 5)))

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
        x = np.random.random((8, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((8, 10, 10, 3))
        generator.fit(x)
        generator = image_generators.ImageFullyConvDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_first')
        # Test grayscale
        x = np.random.random((8, 1, 10, 10))
        generator.fit(x)
        # Test RBG
        x = np.random.random((8, 3, 10, 10))
        generator.fit(x)

    def test_batch_standardize(self):
        # ImageFullyConvDataGenerator.standardize should work on batches
        for test_images in all_test_images():
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
        frames = 7
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batches = images.shape[0] // frames
            images = np.reshape(images, tuple([batches, frames] +
                                              list(images.shape[1:])))
            generator = image_generators.MovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
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
                'X': np.random.random((8, 11, 10, 10, 3)),
                'y': np.random.random((8, 11, 10, 10, 1)),
            }
            generator.flow(train_dict, frames_per_batch=1)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.random(y_shape)
            frames_per_batch = 5
            for x, y in generator.flow(
                    train_dict,
                    shuffle=True,
                    save_to_dir=temp_dir,
                    skip=1,
                    frames_per_batch=frames_per_batch):
                batch_x_shape = tuple([frames_per_batch] + list(images.shape[2:]))
                batch_y_shape = tuple([frames_per_batch] + list(y_shape[2:]))
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(len(y), 2)
                self.assertEqual(y[0].shape[1:], batch_y_shape)
                self.assertEqual(y[-1].shape[1:], batch_y_shape)
                break

    def test_movie_data_generator_channels_first(self):
        frames = 7
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, tuple([batch_count, frames] +
                                              list(images.shape[1:])))
            images = np.rollaxis(images, 4, 1)
            generator = image_generators.MovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 11, 10, 10)),
                'y': np.random.random((8, 1, 11, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.random(y_shape)

            frames_per_batch = 5
            for x, y in generator.flow(
                    train_dict,
                    shuffle=True,
                    skip=1,
                    save_to_dir=temp_dir,
                    frames_per_batch=frames_per_batch):
                batch_x_shape = tuple([images.shape[1], frames_per_batch] +
                                      list(images.shape[3:]))
                batch_y_shape = tuple([y_shape[1], frames_per_batch] +
                                      list(y_shape[3:]))
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(len(y), 2)
                self.assertEqual(y[0].shape[1:], batch_y_shape)
                self.assertEqual(y[-1].shape[1:], batch_y_shape)
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
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((8, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((7, 11, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with bigger frames_per_batch than frames
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((8, 11, 10, 10, 1))
            }
            generator.flow(train_dict, frames_per_batch=31)

        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 3, 10, 10, 5)))

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
        x = np.random.random((8, 5, 10, 10, 1))
        generator.fit(x)
        # Test RBG
        x = np.random.random((8, 5, 10, 10, 3))
        generator.fit(x)
        generator = image_generators.MovieDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=False,
            data_format='channels_first')
        # Test grayscale
        x = np.random.random((8, 1, 5, 4, 6))
        generator.fit(x)
        # Test RBG
        x = np.random.random((8, 3, 5, 4, 6))
        generator.fit(x)

    def test_batch_standardize(self):
        # MovieDataGenerator.standardize should work on batches
        frames = 3
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batches = images.shape[0] // frames
            images = np.reshape(images, tuple([batches, frames] +
                                              list(images.shape[1:])))
            generator = image_generators.MovieDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
                rescale=2,
                preprocessing_function=lambda x: x,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=0.,
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


class TestSiamsesDataGenerator(test.TestCase):

    def _get_dummy_tracking_data(self, length=32, frames=3, batches=8,
                                 channels=1, data_format='channels_last'):
        if data_format == 'channels_last':
            channel_axis = 4
        else:
            channel_axis = 1

        y = []
        while len(y) < frames:
            _y = sk.measure.label(sk.data.binary_blobs(length=length, n_dim=2))
            if len(np.unique(_y)) > 2:
                y.append(_y)

        # remove cell 1 from last frame
        y[-1] = np.where(y[-1] == 1, len(np.unique(y[-1])), y[-1])

        y = np.stack(y, axis=0)  # expand to 3D

        y = np.expand_dims(np.expand_dims(y, axis=0), axis=channel_axis)
        shape = list(y.shape)
        shape[channel_axis] = channels
        x = np.random.random(size=shape)

        return x.astype('float32'), y.astype('int32')

    def test_siamese_data_generator(self):
        frames = 5
        batches = 8

        for channel in [1, 3]:
            images, labels = self._get_dummy_tracking_data(
                32, frames, batches, channel, 'channels_last')

            generator = image_generators.SiameseDataGenerator(
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

            feats = ['appearance', 'distance', 'neighborhood', 'regionprop']

            # Basic test before fit
            train_dict = {
                'X': images,
                'y': labels,
                'daughters': [{1: [2, 3]} for k in range(batches)]
            }
            generator.flow(train_dict, features=feats)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()
            iterator = generator.flow(
                train_dict,
                features=feats,
                crop_dim=2,
                save_to_dir=temp_dir,
                shuffle=True)

            for x, y in iterator:
                assert y['classification'].shape[-1] == 3
                for f in feats:
                    f1 = x['{}_input1'.format(f)]
                    f2 = x['{}_input2'.format(f)]

                    shape1, shape2 = iterator._compute_feature_shape(
                        f, [None] * y['classification'].shape[0])
                    assert f1.shape[1:] == shape1[1:]
                    assert f2.shape[1:] == shape2[1:]
                break

    def test_siamese_data_generator_channels_first(self):
        frames = 5
        batches = 8

        for channel in [1, 3]:
            images, labels = self._get_dummy_tracking_data(
                32, frames, batches, channel, 'channels_first')

            generator = image_generators.SiameseDataGenerator(
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            feats = ['appearance', 'distance', 'neighborhood', 'regionprop']

            # Basic test before fit
            train_dict = {
                'X': images,
                'y': labels,
                'daughters': [{1: [2, 3]} for k in range(batches)]
            }
            generator.flow(train_dict, features=feats)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()
            iterator = generator.flow(
                train_dict,
                features=feats,
                crop_dim=2,
                save_to_dir=temp_dir,
                shuffle=True)

            for x, y in iterator:
                assert y['classification'].shape[-1] == 3
                for f in feats:
                    f1 = x['{}_input1'.format(f)]
                    f2 = x['{}_input2'.format(f)]

                    shape1, shape2 = iterator._compute_feature_shape(
                        f, [None] * y['classification'].shape[0])
                    assert f1.shape[1:] == shape1[1:]
                    assert f2.shape[1:] == shape2[1:]
                break

    def test_siamese_data_generator_invalid_data(self):
        generator = image_generators.SiameseDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last')

        feats = ['appearance', 'distance', 'neighborhood', 'regionprop']

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((8, 10, 10)),
                'y': np.random.random((8, 10, 10)),
                'daughters': {}
            }
            generator.flow(train_dict, features=feats)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((7, 11, 10, 10, 1)),
                'daughters': {}
            }
            generator.flow(train_dict, features=feats)
        # Test flow without daughters
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((7, 11, 10, 10, 1)),
            }
            generator.flow(train_dict, features=feats)
        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 10, 10, 5)))

        with self.assertRaises(ValueError):
            generator = image_generators.SiameseDataGenerator(
                data_format='unknown')

        generator = image_generators.SiameseDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.SiameseDataGenerator(
                zoom_range=(2, 2, 2))


class TestScaleDataGenerator(test.TestCase):

    def test_scale_data_generator(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.ScaleDataGenerator(
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
                'X': np.random.random((8, 10, 10, 3)),
                'y': np.random.randint(0, 9, size=(8, 1)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=(images.shape[0], 1))
            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], (1,))
                break

    def test_scale_data_generator_channels_first(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.ScaleDataGenerator(
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 10, 10)),
                'y': np.random.random((8, 1, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)

            for x, y in generator.flow(
                    train_dict,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                self.assertEqual(y.shape[1:], (1,))
                break

    def test_scale_data_generator_invalid_data(self):
        generator = image_generators.ScaleDataGenerator(
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
                'X': np.random.random((8, 10, 10)),
                'y': np.random.random((8, 10, 10))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((7, 10, 10, 1))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 10, 10, 5)))

        with self.assertRaises(ValueError):
            generator = image_generators.ScaleDataGenerator(
                data_format='unknown')

        generator = image_generators.ScaleDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.ScaleDataGenerator(
                zoom_range=(2, 2, 2))


class TestSemanticDataGenerator(test.TestCase):
    def test_semantic_data_generator(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.SemanticDataGenerator(
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
                'X': np.random.random((8, 10, 10, 3)),
                'y': np.random.random((8, 10, 10, 1)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(len(y), len(transforms))
                self.assertEqual(x.shape[1:], images.shape[1:])
                break

    def test_semantic_data_generator_channels_first(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.SemanticDataGenerator(
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 10, 10)),
                'y': np.random.random((8, 1, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(len(y), len(transforms))
                self.assertEqual(x.shape[1:], images.shape[1:])
                break

    def test_semantic_data_generator_multiple_labels(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            generator = image_generators.SemanticDataGenerator(
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
                'X': np.random.random((8, 10, 10, 3)),
                'y': np.random.random((8, 10, 10, 2)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()
            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [2])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['watershed-cont', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(len(y), len(transforms) * y_shape[-1])
                self.assertEqual(x.shape[1:], images.shape[1:])
                break

    def test_semantic_data_generator_multiple_labels_channels_first(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            generator = image_generators.SemanticDataGenerator(
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 10, 10)),
                'y': np.random.random((8, 2, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0], 2] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['watershed-cont', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(len(y), len(transforms) * y_shape[1])
                self.assertEqual(x.shape[1:], images.shape[1:])
                break

    def test_semantic_data_generator_invalid_data(self):
        generator = image_generators.SemanticDataGenerator(
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
                'X': np.random.random((8, 10, 10)),
                'y': np.random.random((8, 10, 10))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((7, 10, 10, 1))
            }
            generator.flow(train_dict)
        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 10, 10, 5)))

        with self.assertRaises(ValueError):
            generator = image_generators.SemanticDataGenerator(
                data_format='unknown')

        generator = image_generators.SemanticDataGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.SemanticDataGenerator(
                zoom_range=(2, 2, 2))


class TestCroppingDataGenerator(test.TestCase):

    def test_cropping_data_generator(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            crop_size = (17, 17)
            generator = image_generators.CroppingDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
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
                rescale=2,
                crop_size=crop_size)

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 21, 21, 3)),
                'y': np.random.random((8, 21, 21, 1)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(y[0].shape[1:3], crop_size)
                self.assertEqual(x.shape[1:3], crop_size)
                break

            # test with no cropping
            generator = image_generators.CroppingDataGenerator(
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
                rescale=2,
                crop_size=None)

            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(y[0].shape[1:3], train_dict['y'].shape[1:3])
                self.assertEqual(x.shape[1:3], train_dict['X'].shape[1:3])
                break

            # test cropsize=image_size
            generator = image_generators.CroppingDataGenerator(
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
                rescale=2,
                crop_size=(20, 20))

            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(y[0].shape[1:3], train_dict['y'].shape[1:3])
                self.assertEqual(x.shape[1:3], train_dict['X'].shape[1:3])
                break

    def test_cropping_data_generator_channels_first(self):
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                img_list.append(img_to_array(im)[None, ...])

            images = np.vstack(img_list)
            images = np.rollaxis(images, 3, 1)
            crop_size = (17, 17)
            generator = image_generators.CroppingDataGenerator(
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                rescale=2,
                crop_size=crop_size,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 21, 21)),
                'y': np.random.random((8, 1, 21, 21)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            y_shape = tuple([images.shape[0]] + [1] + list(images.shape[2:4]))
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(y[0].shape[2:4], crop_size)
                self.assertEqual(x.shape[2:4], crop_size)
                break

            # test with no cropping
            generator = image_generators.CroppingDataGenerator(
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                rescale=2,
                crop_size=None,
                data_format='channels_first')

            for x, y in generator.flow(
                    train_dict,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(y[0].shape[2:4], train_dict['y'].shape[2:4])
                self.assertEqual(x.shape[2:4], train_dict['X'].shape[2:4])
                break

    def test_cropping_data_generator_invalid_data(self):
        # invalid data with cropping
        cropping_generator = image_generators.CroppingDataGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=True,
            data_format='channels_last',
            crop_size=(11, 11))

        with self.assertRaises(ValueError):
            # crop is larger than image size
            train_dict = {
                'X': np.random.random((8, 10, 10, 3)),
                'y': np.random.randint(0, 9, size=(8, 10, 10, 1))
            }

            cropping_generator.flow(train_dict).next()

        with self.assertRaises(ValueError):
            # crop is equal to image size on only a single dimension
            train_dict = {
                'X': np.random.random((8, 15, 11, 3)),
                'y': np.random.randint(0, 9, size=(8, 15, 11, 1))
            }

            cropping_generator.flow(train_dict).next()

        with self.assertRaises(ValueError):
            # crop size is not a list/tuple
            cropping_generator = image_generators.CroppingDataGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=True,
                data_format='channels_last',
                crop_size=11)


class TestSemanticMovieGenerator(test.TestCase):

    def test_semantic_movie_generator(self):
        frames = 7
        frames_per_batch = 5
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batches = images.shape[0] // frames
            images = np.reshape(images, tuple([batches, frames] +
                                              list(images.shape[1:])))
            generator = image_generators.SemanticMovieGenerator(
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
                'X': np.random.random((8, 11, 10, 10, 3)),
                'y': np.random.random((8, 11, 10, 10, 1)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            # generator.fit(images, augment=True, seed=1)
            batch_x_shape = tuple([frames_per_batch] + list(images.shape[2:]))
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    frames_per_batch=frames_per_batch,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(len(y), len(transforms))
                break

    def test_semantic_movie_generator_channels_first(self):
        frames = 7
        frames_per_batch = 5
        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, tuple([batch_count, frames] +
                                              list(images.shape[1:])))
            images = np.rollaxis(images, 4, 1)
            generator = image_generators.SemanticMovieGenerator(
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
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 11, 10, 10)),
                'y': np.random.random((8, 1, 11, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            # generator.fit(images, augment=True, seed=1)
            batch_x_shape = tuple([images.shape[1], frames_per_batch] +
                                  list(images.shape[3:]))
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    frames_per_batch=frames_per_batch,
                    transforms=transforms,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], batch_x_shape)
                self.assertEqual(len(y), len(transforms))
                break

    def test_semantic_movie_generator_invalid_data(self):
        generator = image_generators.SemanticMovieGenerator(
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
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((8, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((7, 11, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with bigger frames_per_batch than frames
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((8, 11, 10, 10, 1))
            }
            generator.flow(train_dict, frames_per_batch=31)

        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 3, 10, 10, 5)))

        with self.assertRaises(ValueError):
            generator = image_generators.SemanticMovieGenerator(
                data_format='unknown')

        generator = image_generators.SemanticMovieGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.SemanticMovieGenerator(
                zoom_range=(2, 2, 2))


class TestSemantic3DGenerator(test.TestCase):

    def test_semantic_3d_generator(self):
        frames = 7
        frames_per_batch = 5
        frame_shape = (12, 12, 1)
        output_shape = (frames_per_batch, frame_shape[0], frame_shape[1])
        aug_3d = False
        rotation_3d = 0

        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batches = images.shape[0] // frames
            images = np.reshape(images, tuple([batches, frames] +
                                              list(images.shape[1:])))
            # TODO - figure out why zca whitening isn't working with 3d datagen
            generator = image_generators.Semantic3DGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
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
                'X': np.random.random((8, 11, 10, 10, 3)),
                'y': np.random.random((8, 11, 10, 10, 1)),
            }

            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)

            batch_x_shape = tuple([frames_per_batch] + list(images.shape[2:]))
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    frames_per_batch=frames_per_batch,
                    transforms=transforms,
                    frame_shape=frame_shape,
                    aug_3d=aug_3d,
                    rotation_3d=rotation_3d,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], output_shape + (x.shape[-1],))
                self.assertEqual(len(y), len(transforms))
                break

    def test_semantic_3d_rotation(self):
        frames = 5
        frames_per_batch = 5
        frame_shape = (10, 10, 1)
        z_scale = 2
        output_shape = (frames_per_batch, frame_shape[0], frame_shape[1])
        aug_3d = True
        rotation_3d = 90

        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batches = images.shape[0] // frames
            images = np.reshape(images, tuple([batches, frames] +
                                              list(images.shape[1:])))
            generator = image_generators.Semantic3DGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
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
                'X': np.random.random((8, 11, 10, 10, 3)),
                'y': np.random.random((8, 11, 10, 10, 1)),
            }

            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            batch_x_shape = tuple([frames_per_batch] + list(images.shape[2:]))
            y_shape = tuple(list(images.shape)[:-1] + [1])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    frames_per_batch=frames_per_batch,
                    transforms=transforms,
                    frame_shape=frame_shape,
                    aug_3d=aug_3d,
                    rotation_3d=rotation_3d,
                    z_scale=z_scale,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], output_shape + (x.shape[-1],))
                self.assertEqual(len(y), len(transforms))
                break

    def test_semantic_3d_generator_channels_first(self):
        frames = 7
        frames_per_batch = 5
        frame_shape = (12, 12, 1)
        output_shape = (frames_per_batch, frame_shape[0], frame_shape[1])
        aug_3d = True
        rotation_3d = 0

        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, tuple([batch_count, frames] +
                                              list(images.shape[1:])))
            images = np.rollaxis(images, 4, 1)
            generator = image_generators.Semantic3DGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 11, 10, 10)),
                'y': np.random.random((8, 1, 11, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            batch_x_shape = tuple([images.shape[1], frames_per_batch] +
                                  list(images.shape[3:]))
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    frames_per_batch=frames_per_batch,
                    transforms=transforms,
                    frame_shape=frame_shape,
                    aug_3d=aug_3d,
                    rotation_3d=rotation_3d,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], (x.shape[1],) + output_shape)
                self.assertEqual(len(y), len(transforms))
                break

    def test_semantic_3d_generator_channels_first_rotation(self):
        frames = 5
        frames_per_batch = 5
        frame_shape = (10, 10, 1)
        z_scale = 2
        output_shape = (frames_per_batch, frame_shape[0], frame_shape[1])
        aug_3d = True
        rotation_3d = 90

        for test_images in all_test_images():
            img_list = []
            for im in test_images:
                frame_list = []
                for _ in range(frames):
                    frame_list.append(img_to_array(im)[None, ...])
                img_stack = np.vstack(frame_list)
                img_list.append(img_stack)

            images = np.vstack(img_list)
            batch_count = images.shape[0] // frames
            images = np.reshape(images, tuple([batch_count, frames] +
                                              list(images.shape[1:])))
            images = np.rollaxis(images, 4, 1)
            generator = image_generators.Semantic3DGenerator(
                featurewise_center=True,
                samplewise_center=True,
                featurewise_std_normalization=True,
                samplewise_std_normalization=True,
                zca_whitening=False,
                rotation_range=90.,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.5,
                zoom_range=0.2,
                channel_shift_range=1.,
                # brightness_range=(1, 5),  # TODO: `channels_first` conflict
                fill_mode='nearest',
                cval=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first')

            # Basic test before fit
            train_dict = {
                'X': np.random.random((8, 3, 11, 10, 10)),
                'y': np.random.random((8, 1, 11, 10, 10)),
            }
            generator.flow(train_dict)

            # Temp dir to save generated images
            temp_dir = self.get_temp_dir()

            # Fit
            generator.fit(images, augment=True, seed=1)
            batch_x_shape = tuple([images.shape[1], frames_per_batch] +
                                  list(images.shape[3:]))
            y_shape = tuple([images.shape[0], 1] + list(images.shape)[2:])
            train_dict['X'] = images
            train_dict['y'] = np.random.randint(0, 9, size=y_shape)
            transforms = ['outer-distance', 'fgbg']
            for x, y in generator.flow(
                    train_dict,
                    frames_per_batch=frames_per_batch,
                    transforms=transforms,
                    frame_shape=frame_shape,
                    aug_3d=aug_3d,
                    rotation_3d=rotation_3d,
                    z_scale=z_scale,
                    save_to_dir=temp_dir,
                    shuffle=True):
                self.assertEqual(x.shape[1:], (x.shape[1],) + output_shape)
                self.assertEqual(len(y), len(transforms))
                break

    def test_semantic_3d_generator_invalid_data(self):
        generator = image_generators.Semantic3DGenerator(
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            zca_whitening=False,
            data_format='channels_last')

        # Test fit with invalid data
        with self.assertRaises(ValueError):
            x = np.random.random((3, 10, 10))
            generator.fit(x)

        # Test flow with invalid dimensions
        with self.assertRaises(ValueError):
            train_dict = {
                'X': np.random.random((8, 10, 10, 1)),
                'y': np.random.random((8, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with non-matching batches
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((7, 11, 10, 10, 1))
            }
            generator.flow(train_dict)

        # Test flow with bigger frames_per_batch than frames
        with self.assertRaises(Exception):
            train_dict = {
                'X': np.random.random((8, 11, 10, 10, 1)),
                'y': np.random.random((8, 11, 10, 10, 1))
            }
            generator.flow(train_dict, frames_per_batch=31)

        # Invalid number of channels: will work but raise a warning
        generator.fit(np.random.random((8, 3, 10, 10, 5)))

        with self.assertRaises(ValueError):
            generator = image_generators.Semantic3DGenerator(
                data_format='unknown')

        generator = image_generators.Semantic3DGenerator(
            zoom_range=(2, 2))
        with self.assertRaises(ValueError):
            generator = image_generators.Semantic3DGenerator(
                zoom_range=(2, 2, 2))
