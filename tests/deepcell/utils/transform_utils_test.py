"""
Tests for transform_utils
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import numpy.testing as np_test
from skimage.io import imread
from tensorflow.python.platform import test

from deepcell.utils.transform_utils import to_categorical
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.utils.transform_utils import rotate_array_0
from deepcell.utils.transform_utils import rotate_array_90
from deepcell.utils.transform_utils import rotate_array_180
from deepcell.utils.transform_utils import rotate_array_270


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


class TransformUtilsTest(test.TestCase):
    def test_distance_transform_2d(self):
        mask = np.random.randint(2, size=(300, 300, 1))

        # TODO: questionable test results.  Should it be bin_size - 1?
        bin_size = 3
        distance = distance_transform_2d(mask, bins=bin_size)
        assert np.unique(distance).size == bin_size - 1

        bin_size = 4
        distance = distance_transform_2d(mask, bins=bin_size)
        assert np.unique(distance).size == bin_size - 1

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
            assert one_hot.shape == expected_shape
            # Make sure there are only 0s and 1s
            assert np.array_equal(one_hot, one_hot.astype(bool))
            # Make sure there is exactly one 1 in a row
            assert np.all(one_hot.sum(axis=-1) == 1)
            # Get original labels back from one hots
            assert np.all(np.argmax(one_hot, -1).reshape(label.shape) == label)

    def test_rotate_array_0(self):
        img = _get_image()
        unrotated_image = rotate_array_0(img)
        np_test.assert_array_equal(unrotated_image, img)

    def test_rotate_array_90(self):
        img = _get_image()
        rotated_image = rotate_array_90(img)
        expected_image = np.rot90(img)
        np_test.assert_array_equal(rotated_image, expected_image)

    def test_rotate_array_180(self):
        img = _get_image()
        rotated_image = rotate_array_180(img)
        expected_image = np.rot90(np.rot90(img))
        np_test.assert_array_equal(rotated_image, expected_image)

    def test_rotate_array_270(self):
        img = _get_image()
        rotated_image = rotate_array_270(img)
        expected_image = np.rot90(np.rot90(np.rot90(img)))
        np_test.assert_array_equal(rotated_image, expected_image)

    def test_rotate_array_90_and_180(self):
        img = _get_image()
        rotated_image1 = rotate_array_90(img)
        rotated_image1 = rotate_array_90(rotated_image1)
        rotated_image2 = rotate_array_180(img)
        np_test.assert_array_equal(rotated_image1, rotated_image2)

if __name__ == '__main__':
    test.main()
