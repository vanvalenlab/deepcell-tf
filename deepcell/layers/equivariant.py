# Copyright 2016-2022 The Van Valen Lab at the California Institute of
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


def _l2_norm(r):
    x = r[..., 0]
    y = r[..., 1]

    output = tf.sqrt(x**2 + y**2)

    return output


def _cdist(r):
    r_0 = tf.expand_dims(r, axis=2)
    multiples = [1,1, tf.shape(r)[1], 1]
    r_0 = tf.tile(r_0, multiples)

    r_1 = tf.expand_dims(r, axis=1)
    multiples = [1, tf.shape(r)[1], 1, 1]
    r_1 = tf.tile(r_1, multiples)

    output = r_0 - r_1
    return output


def _theta(r):
    return tf.math.atan2(r[..., 1], r[..., 0])


def euler(theta_ij, input_order, output_order):
    m_in = tf.range(-input_order, input_order+1)
    m_in = tf.expand_dims(m_in, axis=-1)
    m_out = tf.range(-output_order, output_order+1)
    m_out = tf.expand_dims(m_out, axis=0)

    m_diff = m_out-m_in
    m_diff = tf.expand_dims(m_diff, axis=0)
    m_diff = tf.expand_dims(m_diff, axis=0)
    m_diff = tf.expand_dims(m_diff, axis=0)
    m_diff = tf.dtypes.cast(m_diff, dtype=tf.float32)

    theta_ij = tf.expand_dims(theta_ij, axis=-1)
    theta_ij = tf.expand_dims(theta_ij, axis=-1)

    theta_m = m_diff * theta_ij
    euler = tf.math.exp(tf.dtypes.complex(0.0, theta_m))
    return euler


def create_radial_nn(input_order, output_order, r_shape, n_filters=64):
    output_dim = (2*input_order+1)* (2*output_order+1)
    inputs = Input(shape=(r_shape[1], 
                          r_shape[1],
                          1))
    x = Dense(n_filters, activation='relu')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(n_filters, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(output_dim)(x)

    radial_nn = Model(inputs=inputs,
                      outputs=x)
    return radial_nn


def equivariant_kernel(r, radial_nn, input_order, output_order):
    # Compute equivariant order mixing kernel
    r_vec = _cdist(r)
    r_ij = _l2_norm(r_vec)
    r_ij = tf.expand_dims(r_ij, axis=-1)

    theta_ij = _theta(r_vec)

    r_nn = radial_nn(r_ij)
    new_shape = [tf.shape(r_nn)[0],
                 tf.shape(r_nn)[1],
                 tf.shape(r_nn)[2],
                 2*input_order+1,
                 2*output_order+1]
    r_nn = tf.reshape(r_nn, new_shape)
    r_nn = tf.dtypes.cast(r_nn, dtype=tf.complex64)
    euler_factor = euler(theta_ij, input_order, output_order)

    eq_kernel = tf.math.multiply(r_nn, euler_factor)

    return eq_kernel


def create_self_interaction_kernel(self_interaction, input_order,
                                   output_order):
    # Compute self interaction kernel
    return tf.linalg.diag(self_interaction,
                          k=output_order-input_order,
                          num_rows=2*input_order + 1,
                          num_cols=2*output_order + 1)


def complex_relu(x):
    x_ = tf.math.abs(x)
    x_ = tf.math.divide(tf.nn.relu(x_), x_ + 1e-5)
    x_ = tf.dtypes.cast(x_, dtype=tf.complex64)
    return tf.math.multiply(x_, x)


class E2Conv(Layer):
    def __init__(self,
                 n_filters=64,
                 output_order=0,
                 **kwargs):
        super(SE2Conv, self).__init__(**kwargs)

        self.n_filters = n_filters
        self.output_order = output_order

    def build(self, input_shape):
        """
        inputs: (coords, features)
        input features have shape (batch, points, features)
        """

        r_shape = input_shape[0]
        x_shape = input_shape[1]
        a_shape = input_shape[2]

        self.r_shape = r_shape
        self.x_shape = x_shape
        self.a_shape = a_shape

        self.input_dim = x_shape[-1]

        if self.x_shape[2] % 2 == 0:
            raise ValueError('The m-orders of the input tensor must'
                             ' be odd - orders range from -m to m, including 0')

        self.input_order = (self.x_shape[2] - 1) // 2

        self.radial_nn = create_radial_nn(input_order=self.input_order,
                                          output_order=self.output_order,
                                          r_shape=self.r_shape)

        self.self_interaction = self.add_weight(name='self_interaction',
                                                shape=(2*min(self.output_order,
                                                             self.input_order) + 1,),
                                                initializer='random_uniform',
                                                trainable=True)

        self.built = True

    def call(self, inputs):

        r = inputs[0]
        x = inputs[1]
        a = inputs[2]

        x = tf.dtypes.cast(x, dtype=tf.complex64)

        # Remove the diagonal part of the adjacency matrix
        # to remove self interaction
        a_diag = tf.linalg.diag_part(a)
        a = a - tf.linalg.diag(a_diag)
        a = tf.dtypes.cast(a, dtype=tf.complex64)

        # Compute equivariant order mixing kernel
        W_order = equivariant_kernel(r, self.radial_nn,
                                     input_order=self.input_order,
                                     output_order=self.output_order)

        # Compute equivariant message - mix orders and aggregate features
        equivariant_message = tf.einsum('bijmn,bij,bjmc->binc', W_order, a, x)

        # Compute self interaction message
        W_self = create_self_interaction_kernel(self.self_interaction,
                                                self.input_order,
                                                self.output_order)
        W_self = tf.dtypes.cast(W_self, dtype='complex64')
        self_message = tf.einsum('mn,bjmc->bjnc',
                                 W_self,
                                 x)

        y = self_message + equivariant_message
        return y

    def get_config(self):
        config = {
            'n_filters': self.n_filters,
            'output_order': self.output_order
        }
        base_config = super(SE2Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class E2Attention(Layer):
    def __init__(self,
                 output_order=0,
                 **kwargs):
        super(SE2Transformer, self).__init__(**kwargs)

        self.output_order = output_order

    def build(self, input_shape):
        """
        inputs: (coords, features)
        input features have shape (batch, points, orders, features)
        """

        r_shape = input_shape[0]
        x_shape = input_shape[1]
        a_shape = input_shape[2]

        self.r_shape = r_shape
        self.x_shape = x_shape
        self.a_shape = a_shape

        self.input_dim = x_shape[-1]

        if self.x_shape[2] % 2 == 0:
            raise ValueError('The m-orders of the input tensor must'
                             ' be odd - orders range from -m to m, including 0')

        self.input_order = (self.x_shape[2] - 1) // 2

        self.key_radial_nn = create_radial_nn(self.input_order,
                                              self.output_order,
                                              self.r_shape)
        self.value_radial_nn = create_radial_nn(self.input_order,
                                                self.output_order,
                                                self.r_shape)

        self.query_weights = self.add_weight(name='query_self_weights',
                                             shape=(2*min(self.output_order,
                                                          self.input_order) + 1,),
                                             initializer='random_uniform',
                                             trainable=True)

        self.value_self_weights = self.add_weight(name='value_self_weights',
                                                  shape=(2*min(self.output_order,
                                                               self.input_order) + 1,),
                                                  initializer='random_uniform',
                                                  trainable=True)

        self.built = True

    def call(self, inputs):
        r = inputs[0]
        x = inputs[1]
        a = inputs[2]

        # Cast x to complex
        x = tf.dtypes.cast(x, dtype=tf.complex64)

        # Remove the diagonal part of the adjacency matrix
        # to remove self interaction
        a_diag = tf.linalg.diag_part(a)
        a = a - tf.linalg.diag(a_diag)
        a = tf.dtypes.cast(a, dtype=tf.complex64)

        # Compute value
        value_kernel = equivariant_kernel(r, self.value_radial_nn,
                                          self.input_order, self.output_order)
        value = tf.einsum('bijmn,bij,bjmc->bijnc', value_kernel, a, x)

        # Compute key
        key_kernel = equivariant_kernel(r, self.key_radial_nn,
                                        self.input_order, self.output_order)
        key = tf.einsum('bijmn,bjmc->bijnc', key_kernel, x)

        # Compute query
        query_kernel = create_self_interaction_kernel(self.query_weights,
                                                      self.input_order,
                                                      self.output_order)
        query_kernel = tf.dtypes.cast(query_kernel, dtype='complex64')
        query = tf.einsum('mn,bimc->binc', query_kernel, x)
        d = float((2*self.output_order + 1)*self.x_shape[-1])
        query = tf.multiply(query, 1.0 / math.sqrt(d))

        # Compute attention
        query_conj = tf.math.conj(query)
        attn = tf.einsum('bijnc,bjnc,bij->bijn', key, query_conj, a)
        attn = tf.math.abs(attn)
        attn = tf.reduce_sum(attn, axis=-1)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.dtypes.cast(attn, dtype='complex64')

        # Compute output
        value_self_kernel = create_self_interaction_kernel(self.value_self_weights,
                                                           self.input_order,
                                                           self.output_order)

        value_self_kernel = tf.dtypes.cast(value_self_kernel, dtype='complex64')
        self_message = tf.einsum('mn,bimc->binc', value_self_kernel, x)

        neighbor_message = tf.einsum('bij,bij,bijmc->bimc', a, attn, value)

        y = self_message + neighbor_message

        return y

    def get_config(self):
        config = {
            'output_order': self.output_order
        }
        base_config = super(SE2Transformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
