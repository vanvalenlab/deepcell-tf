# Copyright 2016-2021 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/tf-keras-retinanet/LICENSE
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
"""Upsampling layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras.layers import Layer


class TempMerge(Layer):
    """Layer for merging the time dimension of a Tensor.

    Args:
        encoder_dim (int): desired encoder dimension.
    """
    def __init__(self, encoder_dim=64, **kwargs):
        super(TempMerge, self).__init__(**kwargs)
        self.encoder_dim = encoder_dim

    def call(self, inputs):
        a = inputs[0]
        b = inputs[1]
        new_shape = [-1, tf.shape(b)[2], self.encoder_dim]
        return tf.reshape(a, new_shape)


class TempUnmerge(Layer):
    """Layer for unmerging the time dimension of a Tensor.

    Args:
        encoder_dim (int): desired encoder dimension.
    """
    def __init__(self, encoder_dim=64, **kwargs):
        super(TempUnmerge, self).__init__(**kwargs)
        self.encoder_dim = encoder_dim

    def call(self, inputs):
        a = inputs[0]
        b = inputs[1]
        new_shape = [-1, tf.shape(b)[1], tf.shape(b)[2], self.encoder_dim]
        return tf.reshape(a, new_shape)
