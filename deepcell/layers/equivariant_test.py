# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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
"""Tests for the equivariant layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.keras import backend as K
from keras import keras_parameterized
from keras import testing_utils
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.platform import test

from deepcell import layers
from deepcell.layers import equivariant

from scipy.spatial.distance import cdist


def test_l2_norm():
	test_tensor = np.array([[1, 1], [1, 1]])
	l2 = equivariant._l2_norm(test_tensor)
	l2 = l2.numpy()
	
	assert l2 == np.linalg.norm(test_tensor, axis=-1)


def test_cdist():
	r = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
	r = np.transpose(r)
	r_expand = np.expand_dims(r, axis=0)
	dist = equivariant._cdist(r_expand)
	dist = np.linalg.norm(dist[0].numpy(), axis=-1)

	assert dist == cdist(r, metric='euclidean')


def test_theta():
	r = [[1], [1]]
	theta = equivariant._theta(r)
	
	assert theta[0].numpy() == np.arctan(1)


def test_euler():
	theta_ij = [[0, np.pi], [np.pi, 0]]
	theta_ij = np.expand_dims(theta_ij, axis=0)
	euler = equivariant.euler(theta_ij, input_order=2, output_order=3)

	assert euler.numpy().shape == (1, 2, 2, 5, 7)


def test_create_radial_nn():
	radial_nn = equivariant.create_radial_nn(2, 3, 
											 r_shape=(1, 16, 2), 
											 n_filters=64)

	assert radial_nn.outputs[0].shape.to_list() == [None, 16, 16, 5*7]


def test_equivariant_kernel():
	radial_nn = equivariant.create_radial_nn(2, 3, 
											 r_shape=(1, 16, 2), 
											 n_filters=64)
	eq_kernel = equivariant.create_equivariant_kernel(radial_nn, 2, 3)

	assert tf.shape(eq_kernel).to_list() == [1, 16, 16, 5, 7]


def test_create_self_interaction_kernel():
	input_order = 2
	output_order = 3
	self_interaction = np.random.uniform(size=(2*min(output_order,
                                                      input_order) + 1,))
	sik = equivariant.create_self_interaction_kernel(self_interaction, 
													 input_order,
													 output_order)
	sik_diag = np.diag(sik.numpy(), k=output_order - input_order)

	assert sik_diag == self_interaction
	assert sik.numpy().shape == (5, 7)


@keras_parameterized.run_all_keras_modes
class E2ConvTest(keras_parameterized.TestCase):
	def test_E2Conv():


	def test_E2Attention():


if __name__ == '__main__':
    test.main()



