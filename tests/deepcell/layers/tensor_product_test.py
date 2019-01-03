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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell.utils import testing_utils
from deepcell import layers


class TensorProdTest(test.TestCase):

    @tf_test_util.run_in_graph_and_eager_modes()
    def test_tensorproduct(self):
        custom_objects = {'TensorProduct': layers.TensorProduct}

        testing_utils.layer_test(
            layers.TensorProduct,
            kwargs={'output_dim': 3},
            custom_objects=custom_objects,
            input_shape=(3, 2))

        testing_utils.layer_test(
            layers.TensorProduct,
            kwargs={'output_dim': 3},
            custom_objects=custom_objects,
            input_shape=(3, 4, 2))

        testing_utils.layer_test(
            layers.TensorProduct,
            kwargs={'output_dim': 3},
            custom_objects=custom_objects,
            input_shape=(None, None, 2))

        testing_utils.layer_test(
            layers.TensorProduct,
            kwargs={'output_dim': 3},
            custom_objects=custom_objects,
            input_shape=(3, 4, 5, 2))

        testing_utils.layer_test(
            layers.TensorProduct,
            kwargs={'output_dim': 3,
                    'data_format': 'channels_first'},
            custom_objects=custom_objects,
            input_shape=(3, 2, 4, 5))

        # test no bias
        testing_utils.layer_test(
            layers.TensorProduct,
            kwargs={'output_dim': 2,
                    'use_bias': False},
            custom_objects=custom_objects,
            input_shape=(3, 5, 6, 4))

        # test bad input channel
        with self.assertRaises(ValueError):
            testing_utils.layer_test(
                layers.TensorProduct,
                kwargs={'output_dim': 3},
                custom_objects=custom_objects,
                input_shape=(3, 5, 6, None))

    def test_tensorproduct_regularization(self):
        with self.test_session():
            layer = layers.TensorProduct(
                3,
                kernel_regularizer=keras.regularizers.l1(0.01),
                bias_regularizer='l1',
                activity_regularizer='l2',
                name='tensorproduct_reg')
            layer(keras.backend.variable(np.ones((2, 4))))
            self.assertEqual(3, len(layer.losses))

    def test_tensorproduct_constraints(self):
        with self.test_session():
            k_constraint = keras.constraints.max_norm(0.01)
            b_constraint = keras.constraints.max_norm(0.01)
            layer = layers.TensorProduct(
                3,
                kernel_constraint=k_constraint,
                bias_constraint=b_constraint)
            layer(keras.backend.variable(np.ones((2, 4))))
            self.assertEqual(layer.kernel.constraint, k_constraint)
            self.assertEqual(layer.bias.constraint, b_constraint)
