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
"""Layers to noramlize input images for 2D and 3D images"""

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

class ClusterDense(Layer):
	def __init__(self,
				number_of_clusters=10):

		self.number_of_clusters = number_of_clusters
		super(Cluster, self).__init__(
			activity_regularizer=regularizers.get(activity_regularizer), 
			**kwargs)

	def build(self, input_shape):
		input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 2:
            raise ValueError('Inputs should have rank 2, '
                             'received input shape: %s' % input_shape)

        mu_shape = (self.number_of_clusters)
        self.mu = self.add_weight(shape=mu_shape,
        					      initializer=self.kernel_initializer,
        						  name='mu',
        						  regularizer=self.kernel_regularizer,
        						  constriant=self.kernel_constraint)

        self.input_spec = InputSpec(ndim=2, axes={channel_axis: 1})
        self.built = True

	def compute_output_shape(self, input_shape):
		input_shape = tensor_shape.TensorShape(input_shape).as_list()
		output_shape = input_shape[0:-1] + [self.number_of_clusters]
		return tensor_shape.TensorShape(output_shape)

	def call(self, inputs):
		# Expand dimensions  for broad casting
		zi = tf.tile(tf.expand_dims(inputs, axis=-1), [1,1,self.number_of_clusters])
		mu = tf.expand_dims(tf.expand_dims(mu, axis=0), axis=0)
		qij_temp = 1/(1 + (zi - mu)**2)
		qij = qij_temp / tf.reduce_sum(qij_temp, axis=-1)

		outputs = [mu, qij]

		return outputs

	def get_config(self):
		config = {
			'number_of_clusters': self.number_of_clusters,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
		}
