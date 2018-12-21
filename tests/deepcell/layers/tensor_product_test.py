# Copyright 2016-2018 The Van Valen Lab at the California Institute of
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
"""Tests for the tensor product layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell.utils import testing_utils
from deepcell import layers


class TensorProdTest(test.TestCase):

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_tensorprod_2d(self):
        custom_objects = {'TensorProd2D': layers.TensorProd2D}
        with self.test_session(use_gpu=True):
            testing_utils.layer_test(
                layers.TensorProd2D,
                kwargs={'input_dim': (3, 5, 6, 4),
                        'output_dim': 2},
                custom_objects=custom_objects,
                input_shape=(3, 5, 6, 4))
            testing_utils.layer_test(
                layers.TensorProd2D,
                kwargs={'input_dim': (3, 4, 5, 6),
                        'output_dim': 2,
                        'data_format': 'channels_first'},
                custom_objects=custom_objects,
                input_shape=(3, 4, 5, 6))

        # test no bias
        with self.test_session(use_gpu=True):
            testing_utils.layer_test(
                layers.TensorProd2D,
                kwargs={'input_dim': (3, 5, 6, 4),
                        'output_dim': 2,
                        'use_bias': False},
                custom_objects=custom_objects,
                input_shape=(3, 5, 6, 4))

        # test activation
        with self.test_session(use_gpu=True):
            testing_utils.layer_test(
                layers.TensorProd2D,
                kwargs={'input_dim': (3, 5, 6, 4),
                        'activation': 'relu',
                        'output_dim': 2},
                custom_objects=custom_objects,
                input_shape=(3, 5, 6, 4))

        # test bad input
        with self.test_session(use_gpu=True):
            with self.assertRaises(ValueError):
                testing_utils.layer_test(
                    layers.TensorProd2D,
                    kwargs={'input_dim': (3, 5, 6, 4),
                            'output_dim': 2},
                    custom_objects=custom_objects,
                    input_shape=(3, 5, 6, None))
            with self.assertRaises(ValueError):
                testing_utils.layer_test(
                    layers.TensorProd2D,
                    kwargs={'input_dim': (3, 5, 5, 6, 3),
                            'output_dim': 2},
                    custom_objects=custom_objects,
                    input_shape=(3, 5, 5, 6, 3))

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_tensorprod_3d(self):
        custom_objects = {'TensorProd3D': layers.TensorProd3D}
        with self.test_session(use_gpu=True):
            testing_utils.layer_test(
                layers.TensorProd3D,
                kwargs={'input_dim': (3, 11, 12, 10, 4),
                        'output_dim': 2},
                custom_objects=custom_objects,
                input_shape=(3, 11, 12, 10, 4))
            testing_utils.layer_test(
                layers.TensorProd3D,
                kwargs={'input_dim': (3, 4, 11, 12, 10),
                        'output_dim': 2,
                        'data_format': 'channels_first'},
                custom_objects=custom_objects,
                input_shape=(3, 4, 11, 12, 10))

        # test no bias
        with self.test_session(use_gpu=True):
            testing_utils.layer_test(
                layers.TensorProd3D,
                kwargs={'input_dim': (3, 11, 12, 10, 4),
                        'output_dim': 2,
                        'use_bias': False},
                custom_objects=custom_objects,
                input_shape=(3, 11, 12, 10, 4))

        # test activation
        with self.test_session(use_gpu=True):
            testing_utils.layer_test(
                layers.TensorProd3D,
                kwargs={'input_dim': (3, 11, 12, 10, 4),
                        'activation': 'relu',
                        'output_dim': 2},
                custom_objects=custom_objects,
                input_shape=(3, 11, 12, 10, 4))

        # test bad channel input
        with self.test_session(use_gpu=True):
            with self.assertRaises(ValueError):
                testing_utils.layer_test(
                    layers.TensorProd3D,
                    kwargs={'input_dim': (3, 11, 12, 10, 4),
                            'output_dim': 2},
                    custom_objects=custom_objects,
                    input_shape=(3, 11, 12, 10, None))
            with self.assertRaises(ValueError):
                testing_utils.layer_test(
                    layers.TensorProd2D,
                    kwargs={'input_dim': (3, 5, 6, 4),
                            'output_dim': 2},
                    custom_objects=custom_objects,
                    input_shape=(3, 5, 6, 4))
