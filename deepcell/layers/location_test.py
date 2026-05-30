"""Tests for the location layers"""

from keras import testing_utils
from keras import keras_parameterized
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.platform import test

from deepcell import layers


@keras_parameterized.run_all_keras_modes
class LocationTest(keras_parameterized.TestCase):

    def test_location_2d(self):
        with custom_object_scope({'Location2D': layers.Location2D}):
            testing_utils.layer_test(
                layers.Location2D,
                kwargs={'data_format': 'channels_last'},
                input_shape=(3, 5, 6, 4))
            testing_utils.layer_test(
                layers.Location2D,
                kwargs={'in_shape': (4, 5, 6),
                        'data_format': 'channels_first'},
                input_shape=(3, 4, 5, 6))

    def test_location_3d(self):
        with custom_object_scope({'Location3D': layers.Location3D}):
            testing_utils.layer_test(
                layers.Location3D,
                kwargs={'data_format': 'channels_last'},
                input_shape=(3, 11, 12, 10, 4))
            testing_utils.layer_test(
                layers.Location3D,
                kwargs={'in_shape': (4, 11, 12, 10),
                        'data_format': 'channels_first'},
                input_shape=(3, 4, 11, 12, 10))


if __name__ == '__main__':
    test.main()
