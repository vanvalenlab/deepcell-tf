"""
Test dc_helper_functions.py
"""

import os
import sys
import numpy as np
import numpy.testing as np_test
import cv2
import pytest

from tensorflow.python import keras
from tensorflow.python.platform.test import TestCase

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deepcell.dc_helper_functions import get_immediate_subdirs
from deepcell.dc_helper_functions import axis_softmax
from deepcell.dc_helper_functions import rotate_array_0
from deepcell.dc_helper_functions import rotate_array_90
from deepcell.dc_helper_functions import rotate_array_180
from deepcell.dc_helper_functions import rotate_array_270
from deepcell.dc_helper_functions import to_categorical


### Import gloabl resources

# Set global resource locations
test_image_location = "./resources/phase.tif"
rotated_90_location = "./resources/rotated_90.tif"
rotated_180_location = "./resources/rotated_180.tif"
rotated_270_location = "./resources/rotated_270.tif"

# Load images
test_image = cv2.imread( test_image_location, 0 )
rotated_90 = cv2.imread( rotated_90_location, 0 )
rotated_180 = cv2.imread( rotated_180_location, 0 )
rotated_270 = cv2.imread( rotated_270_location, 0 )


### Begin tests

# def test_get_immediate_subdirs():

def test_axis_softmax():
    '''
    Adapted from the Tensorflow test for the softmax layer.
    '''

    def _ref_softmax(values):
        m = np.max(values)
        e = np.exp(values - m)
        return e / np.sum(e)

    test_case = TestCase()

    with test_case.test_session():
        x = keras.backend.placeholder(ndim=2)
        f = keras.backend.function([x], [keras.activations.softmax(x)])
        test_values = np.random.random((2, 5))

        result = f([test_values])[0]
    expected = _ref_softmax(test_values[0])
    test_case.assertAllClose(result[0], expected, rtol=1e-05)

    with test_case.assertRaises(ValueError):
        x = keras.backend.placeholder(ndim=1)
        keras.activations.softmax(x)
    
    # Testing that the axis_softmax function fails when passed a NumPy array.
    with pytest.raises(AttributeError) as e_info:
        axis_softmax( test_image, 0)
    with pytest.raises(AttributeError) as e_info:
        axis_softmax( test_image, 1)
    with pytest.raises(AttributeError) as e_info:
        axis_softmax( test_image )

def test_rotate_array_0():
    unrotated_image = rotate_array_0(test_image)
    np_test.assert_array_equal( unrotated_image, test_image )

def test_rotate_array_90():
    rotated_image = rotate_array_90(test_image)
    np_test.assert_array_equal( rotated_image, rotated_90 )
    
def test_rotate_array_180():
    rotated_image = rotate_array_180(test_image)
    np_test.assert_array_equal( rotated_image, rotated_180 )
    
def test_rotate_array_270():
    rotated_image = rotate_array_270(test_image)
    np_test.assert_array_equal( rotated_image, rotated_270 )

def test_rotate_array_90_and_180():
    rotated_image1 = rotate_array_90(test_image)
    rotated_image1 = rotate_array_90(rotated_image1)
    rotated_image2 = rotate_array_180(test_image)
    np_test.assert_array_equal( rotated_image1, rotated_image2 )

def test_to_categorical():
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

if __name__ == '__main__':
    pytest.main([__file__])
