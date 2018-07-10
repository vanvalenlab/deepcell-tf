"""
image_generators_test.py

Tests for custom image data generators

@author: David Van Valen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test

from deepcell import image_generators


def _generate_test_images():
    img_w = img_h = 20
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


class TestImageGenerators(test.TestCase):

    def test_image_data_generator(self):
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
            x = np.random.random((32, 10, 10, 3))
            generator.flow(x)

            # Fit
            generator.fit(images, augment=True)

            for x, _ in generator.flow(
                    images,
                    np.arange(images.shape[0]),
                    shuffle=True):
                self.assertEqual(x.shape[1:], images.shape[1:])
                break
