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
"""Tests for transform_utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.measure import label
from tensorflow.python.platform import test
from tensorflow.python.keras import backend as K

from deepcell.utils import transform_utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def _generate_test_masks():
    img_w = img_h = 30
    mask_images = []
    for _ in range(8):
        imarray = np.random.randint(2, size=(img_w, img_h, 1))
        mask_images.append(imarray)
    return mask_images


class TransformUtilsTest(test.TestCase):
    def test_pixelwise_transform_2d(self):
        with self.cached_session():
            K.set_image_data_format('channels_last')
            # test single edge class
            for img in _generate_test_masks():
                img = label(img)
                img = np.squeeze(img)
                pw_img = transform_utils.pixelwise_transform(
                    img, data_format=None, separate_edge_classes=False)
                pw_img_dil = transform_utils.pixelwise_transform(
                    img, dilation_radius=1,
                    data_format='channels_last',
                    separate_edge_classes=False)

                self.assertEqual(pw_img.shape[-1], 3)
                self.assertEqual(pw_img_dil.shape[-1], 3)
                assert(np.all(np.equal(pw_img[..., 0] + pw_img[..., 1], img > 0)))
                self.assertGreater(
                    pw_img_dil[..., 0].sum() + pw_img_dil[..., 1].sum(),
                    pw_img[..., 0].sum() + pw_img[..., 1].sum())

            # test separate edge classes
            for img in _generate_test_masks():
                img = label(img)
                img = np.squeeze(img)
                pw_img = transform_utils.pixelwise_transform(
                    img, data_format=None, separate_edge_classes=True)
                pw_img_dil = transform_utils.pixelwise_transform(
                    img, dilation_radius=1,
                    data_format='channels_last',
                    separate_edge_classes=True)

                self.assertEqual(pw_img.shape[-1], 4)
                self.assertEqual(pw_img_dil.shape[-1], 4)
                assert(np.all(np.equal(pw_img[..., 0] + pw_img[..., 1] + pw_img[..., 2], img > 0)))
                self.assertGreater(
                    pw_img_dil[..., 0].sum() + pw_img_dil[..., 1].sum(),
                    pw_img[..., 0].sum() + pw_img[..., 1].sum())

    def test_pixelwise_transform_3d(self):
        frames = 10
        img_list = []
        for im in _generate_test_masks():
            frame_list = []
            for _ in range(frames):
                frame_list.append(label(im))
            img_stack = np.array(frame_list)
            img_list.append(img_stack)

        with self.cached_session():
            K.set_image_data_format('channels_last')
            # test single edge class
            maskstack = np.vstack(img_list)
            batch_count = maskstack.shape[0] // frames
            new_shape = tuple([batch_count, frames] + list(maskstack.shape[1:]))
            maskstack = np.reshape(maskstack, new_shape)

            for i in range(maskstack.shape[0]):
                img = maskstack[i, ...]
                img = np.squeeze(img)
                pw_img = transform_utils.pixelwise_transform(
                    img, data_format=None, separate_edge_classes=False)
                pw_img_dil = transform_utils.pixelwise_transform(
                    img, dilation_radius=2,
                    data_format='channels_last',
                    separate_edge_classes=False)
                self.assertEqual(pw_img.shape[-1], 3)
                self.assertEqual(pw_img_dil.shape[-1], 3)
                assert(np.all(np.equal(pw_img[..., 0] + pw_img[..., 1], img > 0)))
                self.assertGreater(
                    pw_img_dil[..., 0].sum() + pw_img_dil[..., 1].sum(),
                    pw_img[..., 0].sum() + pw_img[..., 1].sum())

            # test separate edge classes
            maskstack = np.vstack(img_list)
            batch_count = maskstack.shape[0] // frames
            new_shape = tuple([batch_count, frames] + list(maskstack.shape[1:]))
            maskstack = np.reshape(maskstack, new_shape)

            for i in range(maskstack.shape[0]):
                img = maskstack[i, ...]
                img = np.squeeze(img)
                pw_img = transform_utils.pixelwise_transform(
                    img, data_format=None, separate_edge_classes=True)
                pw_img_dil = transform_utils.pixelwise_transform(
                    img, dilation_radius=2,
                    data_format='channels_last',
                    separate_edge_classes=True)
                self.assertEqual(pw_img.shape[-1], 4)
                self.assertEqual(pw_img_dil.shape[-1], 4)
                assert(np.all(np.equal(pw_img[..., 0] + pw_img[..., 1] + pw_img[..., 2], img > 0)))
                self.assertGreater(
                    pw_img_dil[..., 0].sum() + pw_img_dil[..., 1].sum(),
                    pw_img[..., 0].sum() + pw_img[..., 1].sum())

    def test_distance_transform_3d(self):
        mask_stack = np.array(_generate_test_masks())
        unique = np.zeros(mask_stack.shape)

        for i, mask in enumerate(_generate_test_masks()):
            unique[i] = label(mask)

        K.set_image_data_format('channels_last')

        bins = 3
        distance = transform_utils.distance_transform_3d(unique, bins=bins)
        distance = np.expand_dims(distance, axis=-1)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
        self.assertEqual(distance.shape, unique.shape)

        bins = 4
        distance = transform_utils.distance_transform_3d(unique, bins=bins)
        distance = np.expand_dims(distance, axis=-1)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
        self.assertEqual(distance.shape, unique.shape)

        K.set_image_data_format('channels_first')
        unique = np.rollaxis(unique, -1, 1)

        bins = 3
        distance = transform_utils.distance_transform_3d(unique, bins=bins)
        distance = np.expand_dims(distance, axis=1)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
        self.assertEqual(distance.shape, unique.shape)

        bins = 4
        distance = transform_utils.distance_transform_3d(unique, bins=bins)
        distance = np.expand_dims(distance, axis=1)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
        self.assertEqual(distance.shape, unique.shape)

    def test_distance_transform_2d(self):
        for img in _generate_test_masks():
            K.set_image_data_format('channels_last')
            bins = 3
            distance = transform_utils.distance_transform_2d(img, bins=bins)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
            self.assertEqual(np.expand_dims(distance, axis=-1).shape, img.shape)

            bins = 4
            distance = transform_utils.distance_transform_2d(img, bins=bins)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
            self.assertEqual(np.expand_dims(distance, axis=-1).shape, img.shape)

            K.set_image_data_format('channels_first')
            img = np.rollaxis(img, -1, 1)

            bins = 3
            distance = transform_utils.distance_transform_2d(img, bins=bins)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
            self.assertEqual(np.expand_dims(distance, axis=1).shape, img.shape)

            bins = 4
            distance = transform_utils.distance_transform_2d(img, bins=bins)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
            self.assertEqual(np.expand_dims(distance, axis=1).shape, img.shape)

    def test_centroid_weighted_distance_transform_2d(self):
        for img in _generate_test_masks():
            K.set_image_data_format('channels_last')
            dist_x, dist_y = \
                transform_utils.centroid_weighted_distance_transform_2d(img)
            self.assertEqual(str(dist_x.dtype), str(K.floatx()))
            self.assertEqual(dist_x.shape, img.shape)
            self.assertEqual(str(dist_y.dtype), str(K.floatx()))
            self.assertEqual(dist_y.shape, img.shape)

    def test_centroid_transform_continuous_2d(self):
        for img in _generate_test_masks():
            K.set_image_data_format('channels_last')
            distance = transform_utils.centroid_transform_continuous_2d(img)
            self.assertEqual(np.expand_dims(distance, axis=-1).shape, img.shape)

            K.set_image_data_format('channels_first')
            img = np.rollaxis(img, -1, 1)

            distance = transform_utils.centroid_transform_continuous_2d(img)
            self.assertEqual(np.expand_dims(distance, axis=1).shape, img.shape)

    def test_centroid_transform_continuous_movie(self):
        # TODO: not sure this is working properly!
        mask_stack = np.array(_generate_test_masks())
        img = np.zeros(mask_stack.shape)

        for i, mask in enumerate(_generate_test_masks()):
            img[i] = label(mask)

        # pylint: disable=E1136
        K.set_image_data_format('channels_last')
        centroids = transform_utils.centroid_transform_continuous_movie(img)
        print(centroids.shape)
        self.assertEqual(str(centroids.dtype), str(K.floatx()))
        self.assertEqual(centroids.shape, img.shape[:-1])

        K.set_image_data_format('channels_first')
        centroids = transform_utils.centroid_transform_continuous_movie(img)
        self.assertEqual(str(centroids.dtype), str(K.floatx()))
        self.assertEqual(centroids.shape, img.shape[:-1])

    def test_distance_transform_continuous_2d(self):
        for img in _generate_test_masks():
            K.set_image_data_format('channels_last')
            distance = transform_utils.distance_transform_continuous_2d(img)
            self.assertEqual(np.expand_dims(distance, axis=-1).shape, img.shape)

            K.set_image_data_format('channels_first')
            img = np.rollaxis(img, -1, 1)

            distance = transform_utils.distance_transform_continuous_2d(img)
            self.assertEqual(np.expand_dims(distance, axis=1).shape, img.shape)

    def test_distance_transform_continuous_movie(self):
        mask_stack = np.array(_generate_test_masks())
        img = np.zeros(mask_stack.shape)

        for i, mask in enumerate(_generate_test_masks()):
            img[i] = label(mask)

        K.set_image_data_format('channels_last')
        distance = transform_utils.distance_transform_continuous_movie(img)
        self.assertEqual(np.expand_dims(distance, axis=-1).shape, img.shape)

        K.set_image_data_format('channels_first')
        img = np.rollaxis(img, -1, 1)

        distance = transform_utils.distance_transform_continuous_movie(img)
        self.assertEqual(np.expand_dims(distance, axis=1).shape, img.shape)

    def test_to_categorical(self):
        num_classes = 5
        shapes = [(1,), (3,), (4, 3), (5, 4, 3), (3, 1), (3, 2, 1)]
        expected_shapes = [(1, num_classes),
                           (3, num_classes),
                           (12, num_classes),
                           (60, num_classes),
                           (3, num_classes),
                           (6, num_classes)]
        labels = [np.random.randint(0, num_classes, shape) for shape in shapes]
        ohes = [transform_utils.to_categorical(l, num_classes) for l in labels]
        for lbl, one_hot, expected_shape in zip(labels, ohes, expected_shapes):
            # Check shape
            self.assertEqual(one_hot.shape, expected_shape)
            # Make sure there are only 0s and 1s
            self.assertAllEqual(one_hot, one_hot.astype(bool))
            # Make sure there is exactly one 1 in a row
            assert np.all(one_hot.sum(axis=-1) == 1)
            # Get original labels back from one hots
            self.assertAllEqual(np.argmax(one_hot, -1).reshape(lbl.shape), lbl)

    def test_rotate_array_0(self):
        img = _get_image()
        unrotated_image = transform_utils.rotate_array_0(img)
        self.assertAllEqual(unrotated_image, img)

    def test_rotate_array_90(self):
        img = _get_image()
        rotated_image = transform_utils.rotate_array_90(img)
        expected_image = np.rot90(img)
        self.assertAllEqual(rotated_image, expected_image)

    def test_rotate_array_180(self):
        img = _get_image()
        rotated_image = transform_utils.rotate_array_180(img)
        expected_image = np.rot90(np.rot90(img))
        self.assertAllEqual(rotated_image, expected_image)

    def test_rotate_array_270(self):
        img = _get_image()
        rotated_image = transform_utils.rotate_array_270(img)
        expected_image = np.rot90(np.rot90(np.rot90(img)))
        self.assertAllEqual(rotated_image, expected_image)

    def test_rotate_array_90_and_180(self):
        img = _get_image()
        rotated_image1 = transform_utils.rotate_array_90(img)
        rotated_image1 = transform_utils.rotate_array_90(rotated_image1)
        rotated_image2 = transform_utils.rotate_array_180(img)
        self.assertAllEqual(rotated_image1, rotated_image2)

if __name__ == '__main__':
    test.main()
