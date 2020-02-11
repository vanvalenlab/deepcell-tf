# Copyright 2016-2019 The Van Valen Lab at the California Institute of
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
"""Layers to created weighted biFPNs"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils

class WeightedAdd(Layer):
	def __init__(self,
				epsilon=1e-4,
				**kwargs):
		self.epsilon = epsilon
		super(WeightedAdd, self).__init__(**kwargs)

	def build(self, input_shape):
		n_in = len(input_shape)
		self.W = self.add_weight(name=self.name,
								shape=(n_in, ),
								initializer = initializers.constant(1/n_in),
								trainable=True,
								dtype = K.floatx())

	def call(self, inputs, **kwargs):
		W = activations.relu(self.W)
		x = tf.reduce_sum([W[i] * inputs[i] for i in range(len(inputs))], axis=0)
		x = x / (tf.reduce_sum(W) + self.epsilon)

	def compute_output_shape(self, input_shape):
		return input_shape[0]

	def get_config(self):
		config = {
			'epsilon': self.epsilon
			}
        base_config = super(WeightedAdd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))