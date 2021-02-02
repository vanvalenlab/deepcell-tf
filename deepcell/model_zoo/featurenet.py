# Copyright 2016-2020 The Van Valen Lab at the California Institute of
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
"""Assortment of CNN architectures for single cell segmentation"""

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


def bn_feature_net_2D(receptive_field=61,
                      input_shape=(256, 256, 1),
                      inputs=None,
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
    """Creates a 2D featurenet.

    Args:
        receptive_field (int): the receptive field of the neural network.
        input_shape (tuple): If no input tensor, create one with this shape.
        inputs (tensor): optional input tensor
        n_features (int): Number of output features
        n_channels (int): number of input channels
        reg (int): regularization value
        n_conv_filters (int): number of convolutional filters
        n_dense_filters (int): number of dense filters
        VGG_mode (bool): If ``multires``, uses ``VGG_mode``
            for multiresolution
        init (str): Method for initalizing weights.
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        location (bool): Whether to include a
            :mod:`deepcell.layers.location.Location2D` layer.
        dilated (bool): Whether to use dilated pooling.
        padding (bool): Whether to use padding.
        padding_mode (str): Type of padding, one of 'reflect' or 'zero'
        multires (bool): Enables multi-resolution mode
        include_top (bool): Whether to include the final layer of the model

    Returns:
        tensorflow.keras.Model: 2D FeatureNet
    """
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

    if inputs is not None:
        if not K.is_keras_tensor(inputs):
            img_input = Input(tensor=inputs, shape=input_shape)
        else:
            img_input = inputs
        x.append(img_input)
    else:
        x.append(Input(shape=input_shape))

    x.append(ImageNormalization2D(norm_method=norm_method,
                                  filter_size=receptive_field)(x[-1]))

    if padding:
        if padding_mode == 'reflect':
            x.append(ReflectionPadding2D(padding=(win, win))(x[-1]))
        elif padding_mode == 'zero':
            x.append(ZeroPadding2D(padding=(win, win))(x[-1]))

    if location:
        x.append(Location2D(in_shape=tuple(x[-1].shape.as_list()[1:]))(x[-1]))
        x.append(Concatenate(axis=channel_axis)([x[-2], x[-1]]))

    layers_to_concat = []

    rf_counter = receptive_field
    block_counter = 0
    d = 1

    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        x.append(Conv2D(n_conv_filters, filter_size, dilation_rate=d,
                        kernel_initializer=init, padding='valid',
                        kernel_regularizer=l2(reg))(x[-1]))
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

            rf_counter = rf_counter // 2

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

        if multires:
            x.append(Concatenate(axis=channel_axis)(c))

    x.append(Conv2D(n_dense_filters, (rf_counter, rf_counter), dilation_rate=d,
                    kernel_initializer=init, padding='valid',
                    kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    if include_top:
        x.append(TensorProduct(n_dense_filters, kernel_initializer=init,
                               kernel_regularizer=l2(reg))(x[-1]))
        x.append(BatchNormalization(axis=channel_axis)(x[-1]))
        x.append(Activation('relu')(x[-1]))

        x.append(TensorProduct(n_features, kernel_initializer=init,
                               kernel_regularizer=l2(reg))(x[-1]))

        if not dilated:
            x.append(Flatten()(x[-1]))

        x.append(Softmax(axis=channel_axis, dtype=K.floatx())(x[-1]))

    if inputs is not None:
        real_inputs = keras_utils.get_source_inputs(x[0])
    else:
        real_inputs = x[0]

    model = Model(inputs=real_inputs, outputs=x[-1])

    return model


def bn_feature_net_skip_2D(receptive_field=61,
                           input_shape=(256, 256, 1),
                           inputs=None,
                           fgbg_model=None,
                           n_skips=2,
                           last_only=True,
                           norm_method='std',
                           padding_mode='reflect',
                           **kwargs):
    """Creates a 2D featurenet with skip-connections.

    Args:
        receptive_field (int): the receptive field of the neural network.
        input_shape (tuple): If no input tensor, create one with this shape.
        inputs (tensor): optional input tensor
        fgbg_model (tensorflow.keras.Model): Concatenate output of this model
            with the inputs as a skip-connection.
        last_only (bool): Model will only output the final prediction,
            and not return any of the underlying model predictions.
        n_skips (int): The number of skip-connections
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        padding_mode (str): Type of padding, one of 'reflect' or 'zero'
        kwargs (dict): Other model options defined in `~bn_feature_net_2D`

    Returns:
        tensorflow.keras.Model: 2D FeatureNet with skip-connections
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    inputs = Input(shape=input_shape)
    img = ImageNormalization2D(norm_method=norm_method,
                               filter_size=receptive_field)(inputs)

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
        models.append(bn_feature_net_2D(receptive_field=receptive_field,
                                        input_shape=new_input_shape,
                                        norm_method=None,
                                        dilated=True,
                                        padding=True,
                                        padding_mode=padding_mode,
                                        **kwargs))
        model_outputs.append(models[-1](model_input))

    if last_only:
        model = Model(inputs=inputs, outputs=model_outputs[-1])
    elif fgbg_model is None:
        model = Model(inputs=inputs, outputs=model_outputs)
    else:
        model = Model(inputs=inputs, outputs=model_outputs[1:])

    return model


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
                      include_top=True,
                      temporal=None,
                      residual=False,
                      temporal_kernel_size=3):
    """Creates a 3D featurenet.

    Args:
        receptive_field (int): the receptive field of the neural network.
        n_frames (int): Number of frames.
        input_shape (tuple): If no input tensor, create one with this shape.
        n_features (int): Number of output features
        n_channels (int): number of input channels
        reg (int): regularization value
        n_conv_filters (int): number of convolutional filters
        n_dense_filters (int): number of dense filters
        VGG_mode (bool): If ``multires``, uses ``VGG_mode``
            for multiresolution
        init (str): Method for initalizing weights.
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization3D` layer.
        location (bool): Whether to include a
            :mod:`deepcell.layers.location.Location3D` layer.
        dilated (bool): Whether to use dilated pooling.
        padding (bool): Whether to use padding.
        padding_mode (str): Type of padding, one of 'reflect' or 'zero'
        multires (bool): Enables multi-resolution mode
        include_top (bool): Whether to include the final layer of the model
        temporal (str): Type of temporal operation
        residual (bool): Whether to use temporal information as a residual
        temporal_kernel_size (int): size of 2D kernel used in temporal convolutions

    Returns:
        tensorflow.keras.Model: 3D FeatureNet
    """
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
    x.append(ImageNormalization3D(norm_method=norm_method,
                                  filter_size=receptive_field)(x[-1]))

    if padding:
        if padding_mode == 'reflect':
            x.append(ReflectionPadding3D(padding=(win_z, win, win))(x[-1]))
        elif padding_mode == 'zero':
            x.append(ZeroPadding3D(padding=(win_z, win, win))(x[-1]))

    if location:
        x.append(Location3D(in_shape=tuple(x[-1].shape.as_list()[1:]))(x[-1]))
        x.append(Concatenate(axis=channel_axis)([x[-2], x[-1]]))

    layers_to_concat = []

    rf_counter = receptive_field
    block_counter = 0
    d = 1

    while rf_counter > 4:
        filter_size = 3 if rf_counter % 2 == 0 else 4
        x.append(Conv3D(n_conv_filters, (1, filter_size, filter_size),
                        dilation_rate=(1, d, d), kernel_initializer=init,
                        padding='valid', kernel_regularizer=l2(reg))(x[-1]))
        x.append(BatchNormalization(axis=channel_axis)(x[-1]))
        x.append(Activation('relu')(x[-1]))

        block_counter += 1
        rf_counter -= filter_size - 1

        if block_counter % 2 == 0:
            if dilated:
                x.append(DilatedMaxPool3D(dilation_rate=(1, d, d),
                                          pool_size=(1, 2, 2))(x[-1]))
                d *= 2
            else:
                x.append(MaxPool3D(pool_size=(1, 2, 2))(x[-1]))

            if VGG_mode:
                n_conv_filters *= 2

            rf_counter = rf_counter // 2

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

    x.append(Conv3D(n_dense_filters, (1, rf_counter, rf_counter),
                    dilation_rate=(1, d, d), kernel_initializer=init,
                    padding='valid', kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(Conv3D(n_dense_filters, (n_frames, 1, 1), dilation_rate=(1, d, d),
                    kernel_initializer=init, padding='valid',
                    kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    feature = Activation('relu')(x[-1])

    def __merge_temporal_features(feature, mode='conv', residual=False, n_filters=256,
                                  n_frames=3, padding=True, temporal_kernel_size=3):
        if mode is None:
            return feature

        mode = str(mode).lower()
        if mode == 'conv':
            x = Conv3D(n_filters, (n_frames, temporal_kernel_size, temporal_kernel_size),
                       kernel_initializer=init, padding='same', activation='relu',
                       kernel_regularizer=l2(reg))(feature)
        elif mode == 'lstm':
            x = ConvLSTM2D(filters=n_filters, kernel_size=temporal_kernel_size,
                           padding='same', kernel_initializer=init, activation='relu',
                           kernel_regularizer=l2(reg), return_sequences=True)(feature)
        elif mode == 'gru':
            x = ConvGRU2D(filters=n_filters, kernel_size=temporal_kernel_size,
                          padding='same', kernel_initializer=init, activation='relu',
                          kernel_regularizer=l2(reg), return_sequences=True)(feature)
        else:
            raise ValueError('`temporal` must be one of "conv", "lstm", "gru" or None')

        if residual is True:
            temporal_feature = Add()([feature, x])
        else:
            temporal_feature = x
        temporal_feature_normed = BatchNormalization(axis=channel_axis)(temporal_feature)
        return temporal_feature_normed

    temporal_feature = __merge_temporal_features(feature, mode=temporal, residual=residual,
                                                 n_filters=n_dense_filters, n_frames=n_frames,
                                                 padding=padding,
                                                 temporal_kernel_size=temporal_kernel_size)
    x.append(temporal_feature)

    x.append(TensorProduct(n_dense_filters, kernel_initializer=init,
                           kernel_regularizer=l2(reg))(x[-1]))
    x.append(BatchNormalization(axis=channel_axis)(x[-1]))
    x.append(Activation('relu')(x[-1]))

    x.append(TensorProduct(n_features, kernel_initializer=init,
                           kernel_regularizer=l2(reg))(x[-1]))

    if not dilated:
        x.append(Flatten()(x[-1]))

    if include_top:
        x.append(Softmax(axis=channel_axis, dtype=K.floatx())(x[-1]))

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
    """Creates a 3D featurenet with skip-connections.

    Args:
        receptive_field (int): the receptive field of the neural network.
        input_shape (tuple): Create input tensor with this shape.
        fgbg_model (tensorflow.keras.Model): Concatenate output of this model
            with the inputs as a skip-connection.
        last_only (bool): Model will only output the final prediction,
            and not return any of the underlying model predictions.
        n_skips (int): The number of skip-connections
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization3D` layer.
        padding_mode (str): Type of padding, one of 'reflect' or 'zero'
        kwargs (dict): Other model options defined in `~bn_feature_net_3D`

    Returns:
        tensorflow.keras.Model: 3D FeatureNet with skip-connections
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    inputs = Input(shape=input_shape)
    img = ImageNormalization3D(norm_method=norm_method,
                               filter_size=receptive_field)(inputs)

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
        models.append(bn_feature_net_3D(receptive_field=receptive_field,
                                        input_shape=new_input_shape,
                                        norm_method=None,
                                        dilated=True,
                                        padding=True,
                                        padding_mode=padding_mode,
                                        **kwargs))
        model_outputs.append(models[-1](model_input))

    if last_only:
        model = Model(inputs=inputs, outputs=model_outputs[-1])
    elif fgbg_model is None:
        model = Model(inputs=inputs, outputs=model_outputs)
    else:
        model = Model(inputs=inputs, outputs=model_outputs[1:])

    return model


"""
Tracking Model
"""


def siamese_model(input_shape=None,
                  track_length=1,
                  features=None,
                  neighborhood_scale_size=10,
                  reg=1e-5,
                  init='he_normal',
                  softmax=True,
                  norm_method='std',
                  filter_size=61):
    def compute_input_shape(feature):
        if feature == 'appearance':
            return input_shape
        elif feature == 'distance':
            return (None, 2)
        elif feature == 'neighborhood':
            return (None, 2 * neighborhood_scale_size + 1,
                    2 * neighborhood_scale_size + 1, 1)
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
                InputLayer(input_shape=(None,
                                        2 * neighborhood_scale_size + 1,
                                        2 * neighborhood_scale_size + 1,
                                        1))
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
        input_shape = tuple([input_shape[0], None] + list(input_shape[1:]))
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
