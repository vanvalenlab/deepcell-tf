"""Tests for the convolutional recurrent layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell.utils import testing_utils
from deepcell import layers

class ConvoluationalRecurrentTest(test.TestCase):
    @tf_test_util.run_in_graph_and_eager_modes()
    def test_convolutional_gru_2d(self):
        kernel_size = (3, 3)
        filters = 3
        custom_objects = {'ConvGRU2D': layers.ConvGRU2D}
        for strides in [(1, 1), (2, 2), None]:
            for dilation_rate in [1, 2, (1, 2)]:
                for padding in ['valid', 'same']:
                    with self.test_session():
                        testing_utils.layer_test(
                            layers.ConvGRU2D,
                            kwargs={'filters': filters,
                                    'strides': strides,
                                    'kernel_size': kernel_size,
                                    'padding': padding,
                                    'dilation_rate': dilation_rate,
                                    'data_format': 'channels_last'},
                            custom_objects=custom_objects,
                            input_shape=(3, 11, 12, 10, 4))
                        testing_utils.layer_test(
                            layers.ConvGRU2D,
                            kwargs={'filters': filters,
                                    'strides': strides,
                                    'kernel_size': kernel_size,
                                    'padding': padding,
                                    'dilation_rate': dilation_rate,
                                    'data_format': 'channels_first'},
                            custom_objects=custom_objects,
                            input_shape=(3, 4, 11, 12, 10))
                        


