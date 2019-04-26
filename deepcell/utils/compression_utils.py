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
"""Functions for compressing convolutional neural networks
Adapted from https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning

@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import os

import numpy as np
import tensorly

from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.optimize import minimize_scalar

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Conv3D
from tensorflow.python.keras.layers import Input, Concatenate, Flatten
from tensorflow.python.keras.layers import MaxPool2D, MaxPool3D
from tensorflow.python.keras.layers import Cropping2D, Cropping3D
from tensorflow.python.keras.layers import Activation, Softmax
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import ZeroPadding2D, ZeroPadding3D
from tensorflow.python.keras.regularizers import l2

from deepcell.layers import DilatedMaxPool2D, DilatedMaxPool3D
from deepcell.layers import ImageNormalization2D, ImageNormalization3D
from deepcell.layers import Location2D, Location3D
from deepcell.layers import ReflectionPadding2D, ReflectionPadding3D
from deepcell.layers import TensorProduct

from deepcell import losses
from deepcell import image_generators
from deepcell.utils import train_utils
from deepcell.utils.data_utils import get_data
from deepcell.utils.train_utils import rate_scheduler

def EVBMF(Y, sigma2=None, H=None):
    """
	Taken from https://github.com/CasvandenBogaard/VBMF/blob/master/VBMF.py

    Implementation of the analytical solution to Empirical Variational Bayes Matrix Factorization.
    This function can be used to calculate the analytical solution to empirical VBMF. 
    This is based on the paper and MatLab code by Nakajima et al.:
    "Global analytic solution of fully-observed variational Bayesian matrix factorization."
    Notes
    -----
        If sigma2 is unspecified, it is estimated by minimizing the free energy.
        If H is unspecified, it is set to the smallest of the sides of the input Y.
    Attributes
    ----------
    Y : numpy-array
        Input matrix that is to be factorized. Y has shape (L,M), where L<=M.
    
    sigma2 : int or None (default=None)
        Variance of the noise on Y.
        
    H : int or None (default = None)
        Maximum rank of the factorized matrices.
        
    Returns
    -------
    U : numpy-array
        Left-singular vectors. 
        
    S : numpy-array
        Diagonal matrix of singular values.
        
    V : numpy-array
        Right-singular vectors.
        
    post : dictionary
        Dictionary containing the computed posterior values.
        
        
    References
    ----------
    .. [1] Nakajima, Shinichi, et al. "Global analytic solution of fully-observed variational Bayesian matrix factorization." Journal of Machine Learning Research 14.Jan (2013): 1-37.
    
    .. [2] Nakajima, Shinichi, et al. "Perfect dimensionality recovery by variational Bayesian PCA." Advances in Neural Information Processing Systems. 2012.     
    """   
    L,M = Y.shape #has to be L<=M

    if H is None:
        H = L

    alpha = L/M
    tauubar = 2.5129*np.sqrt(alpha)
    
    #SVD of the input matrix, max rank of H
    U,s,V = np.linalg.svd(Y)
    U = U[:,:H]
    s = s[:H]
    V = V[:H].T 

    #Calculate residual
    residual = 0.
    if H<L:
        residual = np.sum(np.sum(Y**2)-np.sum(s**2))

    #Estimation of the variance when sigma2 is unspecified
    if sigma2 is None: 
        xubar = (1+tauubar)*(1+alpha/tauubar)
        eH_ub = int(np.min([np.ceil(L/(1+alpha))-1, H]))-1
        upper_bound = (np.sum(s**2)+residual)/(L*M)
        lower_bound = np.max([s[eH_ub+1]**2/(M*xubar), np.mean(s[eH_ub+1:]**2)/M])

        scale = 1.#/lower_bound
        s = s*np.sqrt(scale)
        residual = residual*scale
        lower_bound = lower_bound*scale
        upper_bound = upper_bound*scale

        sigma2_opt = minimize_scalar(EVBsigma2, args=(L,M,s,residual,xubar), bounds=[lower_bound, upper_bound], method='Bounded')
        sigma2 = sigma2_opt.x

        print sigma2

    #Threshold gamma term
    threshold = np.sqrt(M*sigma2*(1+tauubar)*(1+alpha/tauubar))
    pos = np.sum(s>threshold)

    #Formula (15) from [2]
    d = np.multiply(s[:pos]/2, 1-np.divide((L+M)*sigma2, s[:pos]**2) + np.sqrt((1-np.divide((L+M)*sigma2, s[:pos]**2))**2 -4*L*M*sigma2**2/s[:pos]**4) )

    #Computation of the posterior
    post = {}
    post['ma'] = np.zeros(H) 
    post['mb'] = np.zeros(H)
    post['sa2'] = np.zeros(H) 
    post['sb2'] = np.zeros(H) 
    post['cacb'] = np.zeros(H)  

    tau = np.multiply(d, s[:pos])/(M*sigma2)
    delta = np.multiply(np.sqrt(np.divide(M*d, L*s[:pos])), 1+alpha/tau)

    post['ma'][:pos] = np.sqrt(np.multiply(d, delta))
    post['mb'][:pos] = np.sqrt(np.divide(d, delta))
    post['sa2'][:pos] = np.divide(sigma2*delta, s[:pos])
    post['sb2'][:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))
    post['cacb'][:pos] = np.sqrt(np.multiply(d, s[:pos])/(L*M))
    post['sigma2'] = sigma2
    post['F'] = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 
                     + np.sum(M*np.log(tau+1) + L*np.log(tau/alpha +1) - M*tau))

	return U[:,:pos], np.diag(d), V[:,:pos], post

def TuckerConv(Layer):
    def __init__(self, filters,
                 kernel_size,
                 input_weights=None,
                 estimate_rank=None,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TuckerConv, self).__init__(**kwargs)
        self.input_weights = input_weights
        self.rank = len(self.input_weights[0])-2
        self.estimate_rank = estimate_rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.padding = padding

        if input_weights is None:
            raise ValueError('TuckerConv needs to be given weights for the decomposition')            

        if input_weights is not None:
            self.estimate_rank = self._estimate_ranks()
            core, factor_1, factor_2 = self._get_tucker_weights()
            self.core = core
            self.factor_1 = factor_1
            self.factor_2 = factor_2

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=4)

        super(TuckerConv, self).__init__(**kwargs)

	def _estimate_ranks(self):
		""" Unfold the 2 modes of the Tensor the decomposition 
		will be performed on and estimate the ranks by VBMF
		"""
		weights = self.input_weights[0]
        if self.rank == 2:
            unfold_dim_1 = 2
            unfold_dim_2 = 3
        elif self.rank == 3:
            unfold_dim_1 = 3
            unfold_dim_2 = 4

		unfold_1 = tensorly.base.unfold(weights, unfold_dim_1)
		unfold_2 = tensorly.base.unfold(weights, unfold_dim_2)
		_, diag_1, _, _ = VBMF.EVBMF(unfold_1)
		_, diag_2, _, _ = VBMF.EVBMF(unfold_2)
		ranks = [diag_1.shape[0], diag_2.shape[0]]
		return ranks

	def _get_tucker_weights(self):
        """ Get the weights of the Tucker decomposition of
        the input weights
        """
        if self.rank == 2:
            modes = [2,3]
        elif self.rank == 3:
            modes = [3,4]
		ranks = self._estimate_ranks()
        weights = self.input_weights[0]
        core, [factor_1, factor_2] = tensorly.partial_tucker(weights,
            modes=modes, ranks=ranks, init='svd')
        return core, factor_1, factor_2

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        if input_shape[channel_axis] != self.input_weights[-2]:
            raise ValueError('The channel dimension of the inputs '
                                'should be the same as the input channel '
                                'dimension of the weight matrix being compressed')
        input_dim = input_shape[channel_axis]

        # First kernel is a pointwise convolution to decrease channel size
        kernel_shape_1 = (1, 1) + (self.factor_2.shape[0], self.factor_2.shape[1])
        kernel_shape_core = self.kernel_size + (self.core.shape[2], self.core.shape[3])
        kernel_shape_2 = (1, 1) + (self.factor_3.shape[1], self.factor_3.shape[0])

        self.kernel_1 = self.add_weight(shape=kernel_shape_1,
                                      initializer=self.kernel_initializer,
                                      name='kernel_1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel_core = self.add_weight(shape=kernel_shape_core,
                                      initializer=self.kernel_initializer,
                                      name='kernel_core',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel_2 = self.add_weight(shape=kernel_shape_2,
                                      initializer=self.kernel_initializer,
                                      name='kernel_2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        # Set the weights
        kernel_1 = np.expand_dims(np.expand_dims(self.factor_1, axis=0), axis=0)
        if self.rank == 3:
            kernel_1 = np.expand_dims(kernel_1, axis=0)

        kernel_core = self.core
        
        kernel_2 = np.expand_dims(np.expand_dims(self.factor_2.T, axis=0), axis=0)
        if self.rank == 3:
            kernel_2 = np.expand_dims(kernel_2, axis=0)

        bias = self.input_weights[-1] if len(self.input_weights)==2 else None
        weights_to_set = [kernel_1, kernel_core, kernel_2] if bias is None else [kernel_1, kernel_core, kernel_2, bias]
        self.set_weights(weights_to_set)

        # Set input spec.
        if self.rank == 2:
            self.input_spec = InputSpec(ndim=4,
                                    axes={channel_axis: input_dim})
        elif self.rank == 3:
            self.input_spec = InputSpec(ndim=5,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.rank == 2:
            # Compute the first pointwise convolution 
            # to reduce channels 
            output_1 = K.conv2d(inputs, 
                        self.kernel_1,
                        strides=1,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=1)

            # Compute the core convolution 
            output_core = K.conv2d(output_1,
                            self.kernel_core,
                            strides=self.strides,
                            padding=self.padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate) 

            # Compute the final pointwise convolution 
            # to increase channels
            output_2 = K.conv2d(output_1,
                        self.kernel_2,
                        strides=1,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=1)

        elif self.rank == 3:
                        # Compute the first pointwise convolution 
            # to reduce channels 
            output_1 = K.conv3d(inputs, 
                        self.kernel_1,
                        strides=1,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=1)

            # Compute the core convolution 
            output_core = K.conv3d(output_1,
                            self.kernel_core,
                            strides=self.strides,
                            padding=self.pading,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate) 

            # Compute the final pointwise convolution 
            # to increase channels
            output_2 = K.conv3d(output_1,
                        self.kernel_2,
                        strides=1,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=1)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(TuckerConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def SVDTensorProd(Layer):
	def __init__(self,
                 input_dim,
                 output_dim,
                 rank,
 				 input_weights=None,
 				 estimate_rank=False,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SVDTensorProd, self).__init__(**kwargs)
        self.input_weights = input_weights
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.estimate_rank = estimate_rank
        if self.estimate_rank and self.input_weights is not None:
        	self.rank = self._estimate_ranks()
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)

    def _estimate_ranks(self):
        """ Estimate the rank of the weight matrix
        """
    	weights = self.input_weights
    	_, diag, _, _ = EVBMF(weights)
    	rank = diag.shape[0]
    	return rank

	def _get_SVD_weights(self):
        """ Get the SVD of the input weight matrix
        """
		if self.input_weights is not None:
			# Get weights
			if len(self.input_weights) == 1:
				W = self.input_weights
				b = None
			if len(self.input_weights) == 2:
				W, b = self.input_weights
			else:
				raise ValueError('The layer to be compressed can only have '
									'one weight matrix and one bias vector.')

			# Perform SVD on weights
			U, sigma, VT = svd(W)

			# Truncate the weights
			sigma_compressed = sigma[0:self.rank,0:self.rank]
			U_compressed = U[:,0:self.rank]
			VT_compressed = VT[0:self.rank,:]

			# Create weights for new layers
			W_0 = np.matmul(sigma_compressed, VT_compressed)
			W_1 = U_compressed

			return (W_0, W_1, b)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4 or len(input_shape) != 5:
            raise ValueError('Inputs should have rank 4 or 5. Received input shape: ' +
                             str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.ndim = len(input_shape) - 2

        kernel_0_shape = (input_dim, self.rank)
		kernel_1_shape = (self.rank, self.output_dim)

        self.kernel_0 = self.add_weight(
            name='kernel_0',
            shape=kernel_0_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.kernel_1 = self.add_weight(
            name='kernel_1',
            shape=kernel_1_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.input_weights is not None:
        	W_0, W_1, b = self._get_SVD_weights()
        	if b is not None:
	        	self.set_weights([W0, W_1])
	        else:
	        	self.set_weights([W_0, W_1, b])

        # Set input spec.
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.ndim == 2:
            if self.data_format == 'channels_first':
                output = tf.tensordot(inputs, self.kernel_0, axes=[[1], [0]])
                output = tf.tensordot(output, self.kernel_1, axes=[[3], [0]])
                output = tf.transpose(output, perm=[0, 3, 1, 2])

            elif self.data_format == 'channels_last':
                output = tf.tensordot(inputs, self.kernel_0, axes=[[3], [0]])
                output = tf.tensordot(output, self.kernel_1, axes=[[3], [0]])

        elif self.ndim == 3:
            if self.data_format == 'channels_first':
                output = tf.tensordot(inputs, self.kernel_0, axes=[[1], [0]])
                output = tf.tensordot(output, self.kernel_1, axes=[[4], [0]])
                output = tf.transpose(output, perm=[0, 4, 1, 2, 3])

            elif self.data_format == 'channels_last':
                output = tf.tensordot(inputs, self.kernel, axes=[[4], [0]])
                output = tf.tensordot(output, self.kernel_1, axes=[[4], [0]])


        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if len(input_shape) == 4:
            if self.data_format == 'channels_first':
                output_shape = tuple((input_shape[0], 
                                      self.output_dim, 
                                      input_shape[2], 
                                      input_shape[3]))
            else:
                output_shape = tuple((input_shape[0], 
                                      input_shape[1], 
                                      input_shape[2], 
                                      self.output_dim))

        if len(input_shape) == 5:
            if self.data_format == 'channels_first':
                output_shape = tuple((input_shape[0],
                                      self.output_dim,
                                      input_shape[2],
                                      input_shape[3],
                                      input_shape[4]))
            else:
                output_shape = tuple((input_shape[0],
                                      input_shape[1],
                                      input_shape[2],
                                      input_shape[3],
                                      self.output_dim))

        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'data_format': self.data_format,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(SVDTensorProd, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def compress_model(input_model):
    """ Go through layer by layer and compress the convolutional
    and tensor product layers. Batch normalization layers are
    unaffected.
    """

    # Start a new model
    new_model_inputs = []
    new_model_outputs = []
    tensor_dict = {}

    model_output_names = [out.name for out in list(input_model.output)]

    for i, layer in enumerate(input_model.layers):
        # Check the input/outputs for each layer
        input_names = [inp.name for inp in list(layer.input)]
        output_names = [out.name for out in list(layer.output)]

        # Setup model inputs
        if 'input' in layer.name:
            for input_tensor in list(layer.output):
                new_model_inputs.append(input_tsr)
                tensor_dict[input_tensor.name] = input_tensor
            continue

        # Setup layer inputs
        layer_inputs = [tensor_dict[name] for name in input_names]
        if len(layer_inputs) == 1:
            inpt = layer_inputs[-1]
        else:
            inpt = layer_inputs

        # Determine if the layer is a convolutional 
        # or tensor product layer
        if 'conv2d' in layer.name:
            layer_type = 'conv2d'
        elif 'conv3d' in layer.name:
            layer_type = 'conv3d'
        elif 'tensorprod' in layer.name:
            layer_type = 'tensorprod' 
        else:
            layer_type = 'other'

        # Compress the layer using either Tucker
        # decomposition or SVD
        if layer_type == 'conv2d':
            x = TuckerConv(layer.filters, 
                    layer.kernel_size, 
                    input_weights=layer.get_weights(), 
                    dilation_rate=layer.dilation_rate,
                    padding=layer.padding,
                    data_format=layer.data_format,
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                    kernel_initializer=layer.kernel_initializer,
                    bias_initializer=layer.bias_initializer,
                    kernel_regularizer=layer.kernel_regularizer,
                    bias_regularizer=layer.bias_regularizer,
                    activity_regularizer=layer.activity_regularizer,
                    kernel_constraint=layer.kernel_constraint,
                    bias_constraint=layer.bias_constraint)(inpt)
        if layer_type == 'conv3d':
            x = TuckerConv(layer.filters, 
                    layer.kernel_size, 
                    input_weights=layer.get_weights(), 
                    dilation_rate=layer.dilation_rate,
                    padding=layer.padding,
                    data_format=layer.data_format,
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                    kernel_initializer=layer.kernel_initializer,
                    bias_initializer=layer.bias_initializer,
                    kernel_regularizer=layer.kernel_regularizer,
                    bias_regularizer=layer.bias_regularizer,
                    activity_regularizer=layer.activity_regularizer,
                    kernel_constraint=layer.kernel_constraint,
                    bias_constraint=layer.bias_constraint)(inpt)
        if layer_type == 'tensorprod':
            x = SVDTensorProd(layer.input_dim,
                    layer.output_dim,
                    rank,
                    input_weights=layer.get_weights(),
                    estimate_rank=True,
                    data_format=layer.data_format,
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                    kernel_initializer=layer.kernel_initializer,
                    bias_initializer=layer.bias_initializer,
                    kernel_regularizer=layer.kernel_regularizer,
                    bias_regularizer=layer.bias_regularizer,
                    activity_regularizer=layer.activity_regularizer,
                    kernel_constraint=layer.kernel_constraint,
                    bias_constraint=layer.bias_constraint)(inpt)
        if layer_type == 'other':
            x = layer(inpt)   

        # Add the outputs to the tensor dictionary
        for name, output_tensor in zip(output_names, list(x)):
            # Check if this tensor is a model output
            if name in model_output_names:
                new_model_outputs.append(output_tensor)
            tensor_dict[name] = output_tensor

    # Return compressed model
    return Model(new_model_inputs, new_model_outputs)

def fine_tune(compressed_model, data_generator):
"""Fine tune a model after compression

"""

