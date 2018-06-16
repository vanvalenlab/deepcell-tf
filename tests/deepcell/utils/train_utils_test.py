import cv2
import pytest
import numpy as np
import numpy.testing as np_test

from tensorflow.python import keras
from tensorflow.python.platform.test import TestCase

from deepcell.utils.train_utils import axis_softmax

test_image_location = './resources/phase.tif'
test_image = cv2.imread(test_image_location, 0)

def test_axis_softmax():
    """
    Adapted from the Tensorflow test for the softmax layer.
    """

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
        axis_softmax(test_image, 0)
    with pytest.raises(AttributeError) as e_info:
        axis_softmax(test_image, 1)
    with pytest.raises(AttributeError) as e_info:
        axis_softmax(test_image)

if __name__ == '__main__':
    pytest.main([__file__])
