"""Tests for the pooling layers"""

from absl.testing import parameterized

from keras import keras_parameterized
from keras import testing_utils
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell import layers


class DilatedMaxPoolingTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            strides=[(1, 1), (2, 2), None],
            dilation_rate=[1, 2, (1, 2)],
            padding=['valid', 'same']))
    def test_dilated_max_pool_2d(self, strides, dilation_rate, padding):
        pool_size = (3, 3)
        custom_objects = {'DilatedMaxPool2D': layers.DilatedMaxPool2D}
        with self.cached_session():
            with custom_object_scope(custom_objects):
                testing_utils.layer_test(
                    layers.DilatedMaxPool2D,
                    kwargs={'strides': strides,
                            'pool_size': pool_size,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'data_format': 'channels_last'},
                    input_shape=(3, 5, 6, 4))
                testing_utils.layer_test(
                    layers.DilatedMaxPool2D,
                    kwargs={'strides': strides,
                            'pool_size': pool_size,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'data_format': 'channels_first'},
                    input_shape=(3, 4, 5, 6))

    @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
    @parameterized.named_parameters(
        *tf_test_util.generate_combinations_with_testcase_name(
            strides=[1, 2, None],
            dilation_rate=[1, 2, (1, 2, 2)],
            padding=['valid', 'same']))
    def test_dilated_max_pool_3d(self, strides, dilation_rate, padding):
        custom_objects = {'DilatedMaxPool3D': layers.DilatedMaxPool3D}
        pool_size = (3, 3, 3)
        with self.cached_session():
            with custom_object_scope(custom_objects):
                testing_utils.layer_test(
                    layers.DilatedMaxPool3D,
                    kwargs={'strides': strides,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'pool_size': pool_size},
                    input_shape=(3, 11, 12, 10, 4))
                testing_utils.layer_test(
                    layers.DilatedMaxPool3D,
                    kwargs={'strides': strides,
                            'padding': padding,
                            'dilation_rate': dilation_rate,
                            'data_format': 'channels_first',
                            'pool_size': pool_size},
                    input_shape=(3, 4, 11, 12, 10))


if __name__ == '__main__':
    test.main()
