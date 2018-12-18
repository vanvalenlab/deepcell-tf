# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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

from deepcell.utils.transform_utils import to_categorical
from deepcell.utils.transform_utils import deepcell_transform
from deepcell.utils.transform_utils import erode_edges
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.utils.transform_utils import distance_transform_3d
from deepcell.utils.transform_utils import rotate_array_0
from deepcell.utils.transform_utils import rotate_array_90
from deepcell.utils.transform_utils import rotate_array_180
from deepcell.utils.transform_utils import rotate_array_270


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
    def test_deepcell_transform_2d(self):
        maskstack = np.array([label(i) for i in _generate_test_masks()])
        dc_maskstack = deepcell_transform(maskstack, data_format=None)
        dc_maskstack_dilated = deepcell_transform(
            maskstack, dilation_radius=1, data_format='channels_last')

        self.assertEqual(dc_maskstack.shape[-1], 4)
        self.assertEqual(dc_maskstack_dilated.shape[-1], 4)
        self.assertGreater(
            dc_maskstack_dilated[:, :, :, 0].sum() + dc_maskstack_dilated[:, :, :, 1].sum(),
            dc_maskstack[:, :, :, 0].sum() + dc_maskstack[:, :, :, 1].sum())

    def test_deepcell_transform_3d(self):
        frames = 10
        img_list = []
        for im in _generate_test_masks():
            frame_list = []
            for _ in range(frames):
                frame_list.append(label(im))
            img_stack = np.array(frame_list)
            img_list.append(img_stack)

        maskstack = np.vstack(img_list)
        batch_count = maskstack.shape[0] // frames
        maskstack = np.reshape(maskstack, (batch_count, frames, *maskstack.shape[1:]))
        dc_maskstack = deepcell_transform(maskstack, data_format=None)
        dc_maskstack_dilated = deepcell_transform(
            maskstack, dilation_radius=2, data_format='channels_last')
        self.assertEqual(dc_maskstack.shape[-1], 4)
        self.assertEqual(dc_maskstack_dilated.shape[-1], 4)
        self.assertGreater(
            dc_maskstack_dilated[:, :, :, :, 0].sum() + dc_maskstack_dilated[:, :, :, :, 1].sum(),
            dc_maskstack[:, :, :, :, 0].sum() + dc_maskstack[:, :, :, :, 1].sum())

    def test_erode_edges_2d(self):
        for img in _generate_test_masks():
            img = label(img)
            img = np.squeeze(img)

            erode_0 = erode_edges(img, erosion_width=0)
            erode_1 = erode_edges(img, erosion_width=1)
            erode_2 = erode_edges(img, erosion_width=2)

            self.assertEqual(img.shape, erode_0.shape)
            self.assertEqual(erode_0.shape, erode_1.shape)
            self.assertEqual(erode_1.shape, erode_2.shape)
            self.assertAllEqual(erode_0, img)
            self.assertGreater(np.sum(erode_0), np.sum(erode_1))
            self.assertGreater(np.sum(erode_1), np.sum(erode_2))

            # test too few dims
            with self.assertRaises(ValueError):
                erode_1 = erode_edges(img[0, :], erosion_width=1)

    def test_erode_edges_3d(self):
        mask_stack = np.array(_generate_test_masks())
        unique_mask_stack = np.zeros(mask_stack.shape)

        for i, mask in enumerate(_generate_test_masks()):
            unique_mask_stack[i] = label(mask)

        unique_mask_stack = np.squeeze(unique_mask_stack)

        erode_0 = erode_edges(unique_mask_stack, erosion_width=0)
        erode_1 = erode_edges(unique_mask_stack, erosion_width=1)
        erode_2 = erode_edges(unique_mask_stack, erosion_width=2)

        self.assertEqual(unique_mask_stack.shape, erode_0.shape)
        self.assertEqual(erode_0.shape, erode_1.shape)
        self.assertEqual(erode_1.shape, erode_2.shape)
        self.assertAllEqual(erode_0, unique_mask_stack)
        self.assertGreater(np.sum(erode_0), np.sum(erode_1))
        self.assertGreater(np.sum(erode_1), np.sum(erode_2))

        # test too many dims
        with self.assertRaises(ValueError):
            erode_1 = erode_edges(np.expand_dims(unique_mask_stack, axis=-1), erosion_width=1)

    def test_distance_transform_3d(self):
        mask_stack = np.array(_generate_test_masks())
        unique_mask_stack = np.zeros(mask_stack.shape)

        for i, mask in enumerate(_generate_test_masks()):
            unique_mask_stack[i] = label(mask)

        K.set_image_data_format('channels_last')

        bin_size = 3
        distance = distance_transform_3d(unique_mask_stack, bins=bin_size)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
        self.assertEqual(np.expand_dims(distance, axis=-1).shape, unique_mask_stack.shape)

        bin_size = 4
        distance = distance_transform_3d(unique_mask_stack, bins=bin_size)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
        self.assertEqual(np.expand_dims(distance, axis=-1).shape, unique_mask_stack.shape)

        K.set_image_data_format('channels_first')
        unique_mask_stack = np.rollaxis(unique_mask_stack, -1, 1)

        bin_size = 3
        distance = distance_transform_3d(unique_mask_stack, bins=bin_size)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
        self.assertEqual(np.expand_dims(distance, axis=1).shape, unique_mask_stack.shape)

        bin_size = 4
        distance = distance_transform_3d(unique_mask_stack, bins=bin_size)
        self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
        self.assertEqual(np.expand_dims(distance, axis=1).shape, unique_mask_stack.shape)

    def test_distance_transform_2d(self):
        for img in _generate_test_masks():
            K.set_image_data_format('channels_last')
            bin_size = 3
            distance = distance_transform_2d(img, bins=bin_size)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
            self.assertEqual(np.expand_dims(distance, axis=-1).shape, img.shape)

            bin_size = 4
            distance = distance_transform_2d(img, bins=bin_size)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
            self.assertEqual(np.expand_dims(distance, axis=-1).shape, img.shape)

            K.set_image_data_format('channels_first')
            img = np.rollaxis(img, -1, 1)

            bin_size = 3
            distance = distance_transform_2d(img, bins=bin_size)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2]))
            self.assertEqual(np.expand_dims(distance, axis=1).shape, img.shape)

            bin_size = 4
            distance = distance_transform_2d(img, bins=bin_size)
            self.assertAllEqual(np.unique(distance), np.array([0, 1, 2, 3]))
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
        one_hots = [to_categorical(label, num_classes) for label in labels]
        for label, one_hot, expected_shape in zip(labels, one_hots, expected_shapes):
            # Check shape
            self.assertEqual(one_hot.shape, expected_shape)
            # Make sure there are only 0s and 1s
            self.assertAllEqual(one_hot, one_hot.astype(bool))
            # Make sure there is exactly one 1 in a row
            assert np.all(one_hot.sum(axis=-1) == 1)
            # Get original labels back from one hots
            self.assertAllEqual(np.argmax(one_hot, -1).reshape(label.shape), label)

    def test_rotate_array_0(self):
        img = _get_image()
        unrotated_image = rotate_array_0(img)
        self.assertAllEqual(unrotated_image, img)

    def test_rotate_array_90(self):
        img = _get_image()
        rotated_image = rotate_array_90(img)
        expected_image = np.rot90(img)
        self.assertAllEqual(rotated_image, expected_image)

    def test_rotate_array_180(self):
        img = _get_image()
        rotated_image = rotate_array_180(img)
        expected_image = np.rot90(np.rot90(img))
        self.assertAllEqual(rotated_image, expected_image)

    def test_rotate_array_270(self):
        img = _get_image()
        rotated_image = rotate_array_270(img)
        expected_image = np.rot90(np.rot90(np.rot90(img)))
        self.assertAllEqual(rotated_image, expected_image)

    def test_rotate_array_90_and_180(self):
        img = _get_image()
        rotated_image1 = rotate_array_90(img)
        rotated_image1 = rotate_array_90(rotated_image1)
        rotated_image2 = rotate_array_180(img)
        self.assertAllEqual(rotated_image1, rotated_image2)

if __name__ == '__main__':
    test.main()
