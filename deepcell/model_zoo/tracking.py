# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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
"""Assortment of CNN (and GNN) architectures for tracking single cells"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv3D, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Input, Concatenate, InputLayer
from tensorflow.keras.layers import Add, Flatten, Dense, Reshape
from tensorflow.keras.layers import MaxPool2D, MaxPool3D
from tensorflow.keras.layers import Cropping2D, Cropping3D
from tensorflow.keras.layers import Activation, Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D, ZeroPadding3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import utils as keras_utils

from deepcell.layers import ConvGRU2D
from deepcell.layers import DilatedMaxPool2D, DilatedMaxPool3D
from deepcell.layers import ImageNormalization2D, ImageNormalization3D
from deepcell.layers import Location2D, Location3D
from deepcell.layers import ReflectionPadding2D, ReflectionPadding3D
from deepcell.layers import TensorProduct


def siamese_model(input_shape=None,
                  features=None,
                  neighborhood_scale_size=10,
                  reg=1e-5,
                  init='he_normal',
                  filter_size=61):
    """Creates a tracking model based on Siamese Neural Networks(SNNs).

    Args:
        input_shape (tuple): If no input tensor, create one with this shape.
        features (list): Number of output features
        neighborhood_scale_size (int): number of input channels
        reg (int): regularization value
        init (str): Method for initalizing weights
        filter_size (int): the receptive field of the neural network

    Returns:
        tensorflow.keras.Model: 2D FeatureNet
    """
    def compute_input_shape(feature):
        if feature == 'appearance':
            return input_shape
        elif feature == 'distance':
            return (None, 2)
        elif feature == 'neighborhood':
            return (None, 2 * neighborhood_scale_size + 1,
                    2 * neighborhood_scale_size + 1,
                    input_shape[-1])
        elif feature == 'regionprop':
            return (None, 3)
        else:
            raise ValueError('siamese_model.compute_input_shape: '
                             'Unknown feature `{}`'.format(feature))

    def compute_reshape(feature):
        if feature == 'appearance':
            return (64,)
        elif feature == 'distance':
            return (2,)
        elif feature == 'neighborhood':
            return (64,)
        elif feature == 'regionprop':
            return (3,)
        else:
            raise ValueError('siamese_model.compute_output_shape: '
                             'Unknown feature `{}`'.format(feature))

    def compute_feature_extractor(feature, shape):
        if feature == 'appearance':
            # This should not stay: channels_first/last should be used to
            # dictate size (1 works for either right now)
            N_layers = np.int(np.floor(np.log2(input_shape[1])))
            feature_extractor = Sequential()
            feature_extractor.add(InputLayer(input_shape=shape))
            # feature_extractor.add(ImageNormalization2D('std', filter_size=32))
            for layer in range(N_layers):
                feature_extractor.add(Conv3D(64, (1, 3, 3),
                                             kernel_initializer=init,
                                             padding='same',
                                             kernel_regularizer=l2(reg)))
                feature_extractor.add(BatchNormalization(axis=channel_axis))
                feature_extractor.add(Activation('relu'))
                feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))

            feature_extractor.add(Reshape((-1, 64)))
            return feature_extractor

        elif feature == 'distance':
            return None
        elif feature == 'neighborhood':
            N_layers_og = np.int(np.floor(np.log2(2 * neighborhood_scale_size + 1)))
            feature_extractor_neighborhood = Sequential()
            feature_extractor_neighborhood.add(
                InputLayer(input_shape=shape)
            )
            for layer in range(N_layers_og):
                feature_extractor_neighborhood.add(Conv3D(64, (1, 3, 3),
                                                          kernel_initializer=init,
                                                          padding='same',
                                                          kernel_regularizer=l2(reg)))
                feature_extractor_neighborhood.add(BatchNormalization(axis=channel_axis))
                feature_extractor_neighborhood.add(Activation('relu'))
                feature_extractor_neighborhood.add(MaxPool3D(pool_size=(1, 2, 2)))

            feature_extractor_neighborhood.add(Reshape((-1, 64)))

            return feature_extractor_neighborhood
        elif feature == 'regionprop':
            return None
        else:
            raise ValueError('siamese_model.compute_feature_extractor: '
                             'Unknown feature `{}`'.format(feature))

    if features is None:
        raise ValueError('siamese_model: No features specified.')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        raise ValueError('siamese_model: Only channels_last is supported.')
    else:
        channel_axis = -1

    input_shape = tuple([None] + list(input_shape))

    features = sorted(features)

    inputs = []
    outputs = []
    for feature in features:
        in_shape = compute_input_shape(feature)
        re_shape = compute_reshape(feature)
        feature_extractor = compute_feature_extractor(feature, in_shape)

        layer_1 = Input(shape=in_shape, name='{}_input1'.format(feature))
        layer_2 = Input(shape=in_shape, name='{}_input2'.format(feature))

        inputs.extend([layer_1, layer_2])

        # apply feature_extractor if it exists
        if feature_extractor is not None:
            layer_1 = feature_extractor(layer_1)
            layer_2 = feature_extractor(layer_2)

        # LSTM on 'left' side of network since that side takes in stacks of features
        layer_1 = LSTM(64)(layer_1)
        layer_2 = Reshape(re_shape)(layer_2)

        outputs.append([layer_1, layer_2])

    dense_merged = []
    for layer_1, layer_2 in outputs:
        merge = Concatenate(axis=channel_axis)([layer_1, layer_2])
        dense_merge = Dense(128)(merge)
        bn_merge = BatchNormalization(axis=channel_axis)(dense_merge)
        dense_relu = Activation('relu')(bn_merge)
        dense_merged.append(dense_relu)

    # Concatenate outputs from both instances
    merged_outputs = Concatenate(axis=channel_axis)(dense_merged)

    # Add dense layers
    dense1 = Dense(128)(merged_outputs)
    bn1 = BatchNormalization(axis=channel_axis)(dense1)
    relu1 = Activation('relu')(bn1)
    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization(axis=channel_axis)(dense2)
    relu2 = Activation('relu')(bn2)
    dense3 = Dense(3, activation='softmax', name='classification', dtype=K.floatx())(relu2)

    # Instantiate model
    final_layer = dense3
    model = Model(inputs=inputs, outputs=final_layer)

    return model
