# Copyright 2016-2018 David Van Valen at California Institute of Technology
# (Caltech), with support from the Paul Allen Family Foundation, Google,
# & National Institutes of Health (NIH) under Grant U24CA224309-01.
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
"""Assortment of CNN architectures for single cell segmentation
@author: David Van Valen
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Conv3D, ConvLSTM2D, LSTM
from tensorflow.python.keras.layers import Add, Input, Concatenate, Lambda, InputLayer
from tensorflow.python.keras.layers import Flatten, Dense, Reshape
from tensorflow.python.keras.layers import MaxPool2D, MaxPool3D
from tensorflow.python.keras.layers import Cropping2D, Cropping3D
from tensorflow.python.keras.layers import Activation, Softmax
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import ZeroPadding2D, ZeroPadding3D
from tensorflow.python.keras.regularizers import l2

from .layers import DilatedMaxPool2D, DilatedMaxPool3D
from .layers import ImageNormalization2D, ImageNormalization3D
from .layers import Location, Location3D
from .layers import ReflectionPadding2D, ReflectionPadding3D
from .layers import TensorProd2D, TensorProd3D
from .layers import Resize


"""
2D feature nets
"""


def bn_feature_net_2D(receptive_field=61,
                      input_shape=(256, 256, 1),
                      n_features=3,
                      n_channels=1,
                      reg=1e-5,
                      n_conv_filters=64,
                      n_dense_filters=200,
                      VGG_mode=False,
                      init='he_normal',
                      norm_method='std',
                      location=False,
                      dilated=False,
                      padding=False,
                      padding_mode='reflect',
                      multires=False,
                      include_top=True):
    # Create layers list (x) to store all of the layers.
    # We need to use the functional API to enable the multiresolution mode
    x = []

    win = (receptive_field - 1) // 2

    if dilated:
        padding = True

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        row_axis = 2
        col_axis = 3

        if not dilated:
            input_shape = (n_channels, receptive_field, receptive_field)

    else:
        row_axis = 1
        col_axis = 2
        channel_axis = -1
        if not dilated:
            input_shape = (receptive_field, receptive_field, n_channels)

    x.append(Input(shape=input_shape))
    x.append(ImageNormalization2D(norm_method=norm_method, filter_size=receptive_field)(x[-1]))

    if padding:
        if padding_mode == 'reflect':
            x.append(ReflectionPadding2D(padding=(win, win))(x[-1]))
        elif padding_mode == 'zero':
            x.append(ZeroPadding2D(padding=(win, win))(x[-1]))

    if location:
        x.append(Location(in_shape=tuple(x[-1].shape.as_list()[1:]))(x[-1]))
        x.append(Concatenate(axis=channel_axis)([x[-2], x[-1]]))

    if multires:
        layers_to_concat = []

    rf_counter = receptive_field
    block_counter = 0
    d = 1

    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        x.append(Conv2D(n_conv_filters, (filter_size, filter_size), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
        x.append(BatchNormalization(axis=channel_axis)(x[-1]))
        x.append(Activation('relu')(x[-1]))

        block_counter += 1
        rf_counter -= filter_size - 1

        if block_counter % 2 == 0:
            if dilated:
                x.append(DilatedMaxPool2D(dilation_rate=d, pool_size=(2, 2))(x[-1]))
                d *= 2
            else:
                x.append(MaxPool2D(pool_size=(2, 2))(x[-1]))

            if VGG_mode:
                n_conv_filters *= 2

            rf_counter /= 2

            if multires:
                layers_to_concat.append(len(x) - 1)

    if multires:
        c = []
        for l in layers_to_concat:
            output_shape = x[l].get_shape().as_list()
            target_shape = x[-1].get_shape().as_list()

            row_crop = int(output_shape[row_axis] - target_shape[row_axis])
            if row_crop % 2 == 0:
                row_crop = (row_crop // 2, row_crop // 2)
            else:
                row_crop = (row_crop // 2, row_crop // 2 + 1)

            col_crop = int(output_shape[col_axis] - target_shape[col_axis])
            if col_crop % 2 == 0:
                col_crop = (col_crop // 2, col_crop // 2)
            else:
                col_crop = (col_crop // 2, col_crop // 2 + 1)

            cropping = (row_crop, col_crop)

            c.append(Cropping2D(cropping=cropping)(x[l]))
        x.append(Concatenate(axis=channel_axis)(c))

    x.append(Conv2D(n_dense_filters, (rf_counter, rf_counter), dilation_rate=d, kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(TensorProd2D(n_dense_filters, n_dense_filters, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(TensorProd2D(n_dense_filters, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))

    if not dilated:
        x.append(Flatten()(x[-1]))

    if include_top:
        x.append(Softmax(axis=channel_axis)(x[-1]))

    model = Model(inputs=x[0], outputs=x[-1])

    return model


def bn_feature_net_skip_2D(receptive_field=61,
                           input_shape=(256, 256, 1),
                           fgbg_model=None,
                           n_skips=2,
                           last_only=True,
                           norm_method='std',
                           padding_mode='reflect',
                           **kwargs):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    inputs = Input(shape=input_shape)
    img = ImageNormalization2D(norm_method=norm_method, filter_size=receptive_field)(inputs)

    models = []
    model_outputs = []

    if fgbg_model is not None:
        for layer in fgbg_model.layers:
            layer.trainable = False

        models.append(fgbg_model)
        fgbg_output = fgbg_model(inputs)
        if isinstance(fgbg_output, list):
            fgbg_output = fgbg_output[-1]
        model_outputs.append(fgbg_output)

    for _ in range(n_skips + 1):
        if model_outputs:
            model_input = Concatenate(axis=channel_axis)([img, model_outputs[-1]])
        else:
            model_input = img

        new_input_shape = model_input.get_shape().as_list()[1:]
        models.append(bn_feature_net_2D(receptive_field=receptive_field, input_shape=new_input_shape, norm_method=None, dilated=True, padding=True, padding_mode=padding_mode, **kwargs))
        model_outputs.append(models[-1](model_input))

    if last_only:
        model = Model(inputs=inputs, outputs=model_outputs[-1])
    else:
        if fgbg_model is None:
            model = Model(inputs=inputs, outputs=model_outputs)
        else:
            model = Model(inputs=inputs, outputs=model_outputs[1:])

    return model


def bn_feature_net_21x21(**kwargs):
    return bn_feature_net_2D(receptive_field=21, **kwargs)


def bn_feature_net_31x31(**kwargs):
    return bn_feature_net_2D(receptive_field=31, **kwargs)


def bn_feature_net_41x41(**kwargs):
    return bn_feature_net_2D(receptive_field=41, **kwargs)


def bn_feature_net_61x61(**kwargs):
    return bn_feature_net_2D(receptive_field=61, **kwargs)


def bn_feature_net_81x81(**kwargs):
    return bn_feature_net_2D(receptive_field=81, **kwargs)


"""
3D feature nets
"""


def bn_feature_net_3D(receptive_field=61,
                      n_frames=5,
                      input_shape=(5, 256, 256, 1),
                      n_features=3,
                      n_channels=1,
                      reg=1e-5,
                      n_conv_filters=64,
                      n_dense_filters=200,
                      VGG_mode=False,
                      init='he_normal',
                      norm_method='std',
                      location=False,
                      dilated=False,
                      padding=False,
                      padding_mode='reflect',
                      multires=False,
                      include_top=True):
    # Create layers list (x) to store all of the layers.
    # We need to use the functional API to enable the multiresolution mode
    x = []

    win = (receptive_field - 1) // 2
    win_z = (n_frames - 1) // 2

    if dilated:
        padding = True

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        time_axis = 2
        row_axis = 3
        col_axis = 4
        if not dilated:
            input_shape = (n_channels, n_frames, receptive_field, receptive_field)
    else:
        channel_axis = -1
        time_axis = 1
        row_axis = 2
        col_axis = 3
        if not dilated:
            input_shape = (n_frames, receptive_field, receptive_field, n_channels)

    x.append(Input(shape=input_shape))
    x.append(ImageNormalization3D(norm_method=norm_method, filter_size=receptive_field)(x[-1]))

    if padding:
        if padding_mode == 'reflect':
            x.append(ReflectionPadding3D(padding=(win_z, win, win))(x[-1]))
        elif padding_mode == 'zero':
            x.append(ZeroPadding3D(padding=(win_z, win, win))([-1]))

    if location:
        x.append(Location3D(in_shape=tuple(x[-1].shape.as_list()[1:]))(x[-1]))
        x.append(Concatenate(axis=channel_axis)([x[-2], x[-1]]))

    if multires:
        layers_to_concat = []

    rf_counter = receptive_field
    block_counter = 0
    d = 1

    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        x.append(Conv3D(n_conv_filters, (1, filter_size, filter_size), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
        x.append(BatchNormalization(axis=channel_axis)(x[-1]))
        x.append(Activation('relu')(x[-1]))

        block_counter += 1
        rf_counter -= filter_size - 1

        if block_counter % 2 == 0:
            if dilated:
                x.append(DilatedMaxPool3D(dilation_rate=(1, d, d), pool_size=(1, 2, 2))(x[-1]))
                d *= 2
            else:
                x.append(MaxPool3D(pool_size=(1, 2, 2))(x[-1]))

            if VGG_mode:
                n_conv_filters *= 2

            rf_counter /= 2

            if multires:
                layers_to_concat.append(len(x) - 1)

    if multires:
        c = []
        for l in layers_to_concat:
            output_shape = x[l].get_shape().as_list()
            target_shape = x[-1].get_shape().as_list()
            time_crop = (0, 0)

            row_crop = int(output_shape[row_axis] - target_shape[row_axis])

            if row_crop % 2 == 0:
                row_crop = (row_crop // 2, row_crop // 2)
            else:
                row_crop = (row_crop // 2, row_crop // 2 + 1)

            col_crop = int(output_shape[col_axis] - target_shape[col_axis])

            if col_crop % 2 == 0:
                col_crop = (col_crop // 2, col_crop // 2)
            else:
                col_crop = (col_crop // 2, col_crop // 2 + 1)

            cropping = (time_crop, row_crop, col_crop)

            c.append(Cropping3D(cropping=cropping)(x[l]))
        x.append(Concatenate(axis=channel_axis)(c))

    x.append(Conv3D(n_dense_filters, (1, rf_counter, rf_counter), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(Conv3D(n_dense_filters, (n_frames, 1, 1), dilation_rate=(1, d, d), kernel_initializer=init, padding='valid', kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(TensorProd3D(n_dense_filters, n_dense_filters, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(TensorProd3D(n_dense_filters, n_features, kernel_initializer=init, kernel_regularizer=l2(reg))(x[-1]))

    if not dilated:
        x.append(Flatten()(x[-1]))

    if include_top:
        x.append(Softmax(axis=channel_axis)(x[-1]))

    model = Model(inputs=x[0], outputs=x[-1])

    return model


def bn_feature_net_skip_3D(receptive_field=61,
                           input_shape=(5, 256, 256, 1),
                           fgbg_model=None,
                           last_only=True,
                           n_skips=2,
                           norm_method='std',
                           padding_mode='reflect',
                           **kwargs):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    inputs = Input(shape=input_shape)
    img = ImageNormalization3D(norm_method=norm_method, filter_size=receptive_field)(inputs)

    models = []
    model_outputs = []

    if fgbg_model is not None:
        for layer in fgbg_model.layers:
            layer.trainable = False
        models.append(fgbg_model)
        fgbg_output = fgbg_model(inputs)
        if isinstance(fgbg_output, list):
            fgbg_output = fgbg_output[-1]
        model_outputs.append(fgbg_output)

    for _ in range(n_skips + 1):
        if model_outputs:
            model_input = Concatenate(axis=channel_axis)([img, model_outputs[-1]])
        else:
            model_input = img
        new_input_shape = model_input.get_shape().as_list()[1:]
        models.append(bn_feature_net_3D(receptive_field=receptive_field, input_shape=new_input_shape, norm_method=None, dilated=True, padding=True, padding_mode=padding_mode, **kwargs))
        model_outputs.append(models[-1](model_input))

    if last_only:
        model = Model(inputs=inputs, outputs=model_outputs[-1])
    else:
        if fgbg_model is None:
            model = Model(inputs=inputs, outputs=model_outputs)
        else:
            model = Model(inputs=inputs, outputs=model_outputs[1:])

    return model


def bn_feature_net_21x21_3D(**kwargs):
    return bn_feature_net_3D(receptive_field=21, **kwargs)


def bn_feature_net_31x31_3D(**kwargs):
    return bn_feature_net_3D(receptive_field=31, **kwargs)


def bn_feature_net_41x41_3D(**kwargs):
    return bn_feature_net_3D(receptive_field=41, **kwargs)


def bn_feature_net_61x61_3D(**kwargs):
    return bn_feature_net_3D(receptive_field=61, **kwargs)


def bn_feature_net_81x81_3D(**kwargs):
    return bn_feature_net_3D(receptive_field=81, **kwargs)




"""
Tracking Model
"""
import numpy as np

def siamese_model(input_shape=None, track_length=1, occupancy_grid_size=10, reg=1e-5, init='he_normal', softmax=True, norm_method='std', filter_size=61):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        new_input_shape = tuple([input_shape[0]] + [None] + list(input_shape[1:]))
    else:
        channel_axis = -1
        new_input_shape = tuple([None] + list(input_shape))
        
    input_shape = new_input_shape
        
    # Define the input shape for the images
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)
    
    # Define the input shape for the other data (centroids, etc)
    input_3 = Input(shape=(None, 2))
    input_4 = Input(shape=(None, 2))
    
    input_5 = Input(shape=(None, 2*occupancy_grid_size+1, 2*occupancy_grid_size+1, 1))
    input_6 = Input(shape=(None, 2*occupancy_grid_size+1, 2*occupancy_grid_size+1, 1))

    # Feature extractor for images
    N_layers = np.int(np.floor(np.log2(input_shape[1])))  # This should not stay: channels_first/last should be used to dictate size (1 works for either right now)
    feature_extractor = Sequential()
    feature_extractor.add(InputLayer(input_shape=input_shape))
    for layer in range(N_layers):
        feature_extractor.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='same', 
                                     kernel_regularizer=l2(reg)))
        feature_extractor.add(BatchNormalization(axis=channel_axis))
        feature_extractor.add(Activation('relu'))
        feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))

    feature_extractor.add(Reshape(tuple([-1, 64])))
    
    # Feature extractor for occupancy grids
    N_layers_og = np.int(np.floor(np.log2(2*occupancy_grid_size+1)))
    feature_extractor_occupancy_grid = Sequential()
    feature_extractor_occupancy_grid.add(InputLayer(input_shape=(None, 2*occupancy_grid_size+1, 2*occupancy_grid_size+1, 1)))   
    for layer in range(N_layers_og):
        feature_extractor_occupancy_grid.add(Conv3D(64, (1, 3, 3), kernel_initializer=init, padding='same', 
                                                    kernel_regularizer=l2(reg)))
        feature_extractor_occupancy_grid.add(BatchNormalization(axis=channel_axis))
        feature_extractor_occupancy_grid.add(Activation('relu'))
        feature_extractor_occupancy_grid.add(MaxPool3D(pool_size=(1, 2, 2)))
    
    feature_extractor_occupancy_grid.add(Reshape(tuple([-1, 64])))

    # Apply feature extractor to appearances
    output_1 = feature_extractor(input_1)
    output_2 = feature_extractor(input_2)
    
    lstm_1 = LSTM(64)(output_1)
    output_2_reshape = Reshape((64,))(output_2)
    
    # Centroids
    lstm_3 = LSTM(64)(input_3)
    input_4_reshape = Reshape((2,))(input_4)

    # Apply feature extractor to occupancy grids
    output_5 = feature_extractor_occupancy_grid(input_5)
    output_6 = feature_extractor_occupancy_grid(input_6)
    
    lstm_5 = LSTM(64)(output_5)
    output_6_reshape = Reshape((64,))(output_6)
    
    # Combine the extracted features with other known features (centroids)
    merge_1 = Concatenate(axis=channel_axis)([lstm_1, output_2_reshape])
    merge_2 = Concatenate(axis=channel_axis)([lstm_3, input_4_reshape])
    merge_3 = Concatenate(axis=channel_axis)([lstm_5, output_6_reshape])
    
    dense_merge_1 = Dense(128)(merge_1)
    bn_merge_1 = BatchNormalization(axis=channel_axis)(dense_merge_1)
    dense_relu_1 = Activation('relu')(bn_merge_1)
    
    dense_merge_2 = Dense(128)(merge_2)
    bn_merge_2 = BatchNormalization(axis=channel_axis)(dense_merge_2)
    dense_relu_2 = Activation('relu')(bn_merge_2)
    
    dense_merge_3 = Dense(128)(merge_3)
    bn_merge_3 = BatchNormalization(axis=channel_axis)(dense_merge_3)
    dense_relu_3 = Activation('relu')(bn_merge_3)
    
    # Concatenate outputs from both instances
    merged_outputs = Concatenate(axis=channel_axis)([dense_relu_1, dense_relu_2, dense_relu_3])

    # Add dense layers
    dense1 = Dense(128)(merged_outputs)
    bn1 = BatchNormalization(axis=channel_axis)(dense1)
    relu1 = Activation('relu')(bn1)
    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization(axis=channel_axis)(dense2)
    relu2 = Activation('relu')(bn2)
    dense3 = Dense(3, activation='softmax')(relu2)

    # Instantiate model
    final_layer = dense3
    model = Model(inputs=[input_1, input_2, input_3, input_4, input_5, input_6], outputs=final_layer)

    return model