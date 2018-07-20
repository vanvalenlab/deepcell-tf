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
from skimage.measure import label
from tensorflow.python.platform import test
from tensorflow.python.keras import backend as K

from deepcell.utils.transform_utils import to_categorical
from deepcell.utils.transform_utils import distance_transform_2d
from deepcell.utils.transform_utils import distance_transform_3d
from deepcell.utils.transform_utils import rotate_array_0
from deepcell.utils.transform_utils import rotate_array_90
from deepcell.utils.transform_utils import rotate_array_180
from deepcell.utils.transform_utils import rotate_array_270

TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_DIR = os.path.join(TEST_DIR, 'resources')

# Load images
TEST_IMG = imread(os.path.join(RES_DIR, 'phase.tif'))
TEST_MASK = imread(os.path.join(RES_DIR, 'feature_1.tif'))
TEST_IMG_90 = imread(os.path.join(RES_DIR, 'rotated_90.tif'))
TEST_IMG_180 = imread(os.path.join(RES_DIR, 'rotated_180.tif'))
TEST_IMG_270 = imread(os.path.join(RES_DIR, 'rotated_270.tif'))


def _generate_test_masks():
    img_w = img_h = 30
    mask_images = []
    for _ in range(8):
        imarray = np.random.randint(2, size=(img_w, img_h, 1))
        mask_images.append(imarray)
    return mask_images


class TransformUtilsTest(test.TestCase):
    def test_distance_transform_3d(self):
        unique_mask_stack = []
        mask_stack = np.array(_generate_test_masks())
        for mask in _generate_test_masks():
            unique_mask = label(mask)
            unique_mask_stack.append(unique_mask)
        unique_mask_stack = np.array(unique_mask_stack)
        no_channel_mask = np.reshape(mask_stack, unique_mask_stack.shape[:-1])

        K.set_image_data_format('channels_last')
        bin_size = 3
        distance = distance_transform_3d(no_channel_mask, bins=bin_size)
        assert np.array_equal(np.unique(distance), np.array([0, 1, 2]))
        assert distance.shape == no_channel_mask.shape

        bin_size = 4
        distance = distance_transform_3d(no_channel_mask, bins=bin_size)
        assert np.array_equal(np.unique(distance), np.array([0, 1, 2, 3]))
        assert distance.shape == no_channel_mask.shape

        K.set_image_data_format('channels_first')
        bin_size = 3
        distance = distance_transform_3d(no_channel_mask, bins=bin_size)
        assert np.array_equal(np.unique(distance), np.array([0, 1, 2]))
        assert distance.shape == no_channel_mask.shape

        bin_size = 4
        distance = distance_transform_3d(no_channel_mask, bins=bin_size)
        assert np.array_equal(np.unique(distance), np.array([0, 1, 2, 3]))
        assert distance.shape == no_channel_mask.shape

    def test_distance_transform_2d(self):
        for img in _generate_test_masks():
            K.set_image_data_format('channels_last')
            bin_size = 3
            distance = distance_transform_2d(img, bins=bin_size)
            assert np.array_equal(np.unique(distance), np.array([0, 1, 2]))

            bin_size = 4
            distance = distance_transform_2d(img, bins=bin_size)
            assert np.array_equal(np.unique(distance), np.array([0, 1, 2, 3]))

            K.set_image_data_format('channels_first')
            img = np.rollaxis(img, 2, 1)
            bin_size = 3
            distance = distance_transform_2d(img, bins=bin_size)
            assert np.array_equal(np.unique(distance), np.array([0, 1, 2]))

            bin_size = 4
            distance = distance_transform_2d(img, bins=bin_size)
            assert np.array_equal(np.unique(distance), np.array([0, 1, 2, 3]))

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
        unrotated_image = rotate_array_0(TEST_IMG)
        np_test.assert_array_equal(unrotated_image, TEST_IMG)

    def test_rotate_array_90(self):
        rotated_image = rotate_array_90(TEST_IMG)
        np_test.assert_array_equal(rotated_image, TEST_IMG_90)

    def test_rotate_array_180(self):
        rotated_image = rotate_array_180(TEST_IMG)
        np_test.assert_array_equal(rotated_image, TEST_IMG_180)

    def test_rotate_array_270(self):
        rotated_image = rotate_array_270(TEST_IMG)
        np_test.assert_array_equal(rotated_image, TEST_IMG_270)

    def test_rotate_array_90_and_180(self):
        rotated_image1 = rotate_array_90(TEST_IMG)
        rotated_image1 = rotate_array_90(rotated_image1)
        rotated_image2 = rotate_array_180(TEST_IMG)
        np_test.assert_array_equal(rotated_image1, rotated_image2)

if __name__ == '__main__':
    test.main()
