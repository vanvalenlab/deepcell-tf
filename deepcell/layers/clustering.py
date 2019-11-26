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
                number_of_clusters=10,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                **kwargs):

        super(ClusterDense, self).__init__(**kwargs)

        self.number_of_clusters = number_of_clusters
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 2:
            raise ValueError('Inputs should have rank 2, '
                             'received input shape: %s' % input_shape)
        embedding_dim = input_shape[-1]

        mu_shape = (embedding_dim, self.number_of_clusters)
        self.mu = self.add_weight(shape=mu_shape,
                                  initializer=self.kernel_initializer,
                                  name='mu',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)

        self.input_spec = InputSpec(ndim=3)
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        mu_output_shape = [embedding_dim, self.number_of_clusters]
        qij_output_shape = [input_shape[0], self.number_of_clusters]
        return [tensor_shape.TensorShape(mu_output_shape), tensor_shape.TensorShape(qij_output_shape)]

    def call(self, inputs):
        # Expand dimensions  for broad casting
        zi = K.tile(K.expand_dims(inputs, axis=-1), [1,1,self.number_of_clusters])
        mu = K.expand_dims(self.mu, axis=0)
        qij_temp = 1/(1 + K.sum((zi - mu)**2, axis=1))
        qij = qij_temp / K.sum(qij_temp, axis=-1, keepdims=True)

        outputs = [self.mu, qij]

        return outputs

    def get_config(self):
        config = {
            'number_of_clusters': self.number_of_clusters,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }