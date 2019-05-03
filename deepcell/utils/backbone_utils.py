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
"""Functions for creating model backbones"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import applications
from tensorflow.python.keras import utils as keras_utils
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, Conv3D, BatchNormalization
from tensorflow.python.keras.layers import Activation, MaxPool2D, MaxPool3D


def featurenet_block(x, n_filters):
    """Add a set of layers that make up one unit of the featurenet backbone
    Args:
        x (layer): Keras layer object to pass to backbone unit
        n_filters (int): Number of filters to use for convolutional layers
    Returns:
        layer: Keras layer object
    """

    # conv set 1
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # conv set 2
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', data_format='channels_last')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # Final max pooling stage
    x = MaxPool2D(pool_size=(2, 2), data_format='channels_last')(x)

    return x


def featurenet_3D_block(x, n_filters):
    """Add a set of layers that make up one unit of the featurenet 3D backbone
    Args:
        x (layer): Keras layer object to pass to backbone unit
        n_filters (int): Number of filters to use for convolutional layers
    Returns:
        layer: Keras layer object
    """

    # conv set 1
    x = Conv3D(n_filters, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format=K.image_data_format())(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # conv set 2
    x = Conv3D(n_filters, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format=K.image_data_format())(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # Final max pooling stage
    x = MaxPool3D(pool_size=(2, 2, 2), data_format=K.image_data_format())(x)

    return x


def featurenet_backbone(input_tensor=None, input_shape=None, weights=None, include_top=False, pooling=None, n_filters=32, n_dense=128, n_classes=3):
    """Construct the deepcell backbone with five convolutional units
        input_tensor (tensor): Input tensor to specify input size
        n_filters (int, optional): Defaults to 32. Number of filters for convolutionaal layers
    Returns:
        (backbone_names, backbone_features): List of backbone layers, list of backbone names
    """


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Build out backbone
    c1 = featurenet_block(img_input, n_filters)  # 1/2 64x64
    c2 = featurenet_block(c1, n_filters)  # 1/4 32x32
    c3 = featurenet_block(c2, n_filters)  # 1/8 16x16
    c4 = featurenet_block(c3, n_filters)  # 1/16 8x8
    c5 = featurenet_block(c4, n_filters)  # 1/32 4x4

    backbone_features = [c1, c2, c3, c4, c5]
    backbone_names = ['C1', 'C2', 'C3', 'C4', 'C5']
    output_dict = {}
    for name, feature in zip(backbone_names, backbone_features):
        output_dict[name] = feature

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=backbone_features)
    return model, output_dict


def featurenet_3D_backbone(input_tensor=None, input_shape=None, weights=None, include_top=False, pooling=None, n_filters=32, n_dense=128, n_classes=3):
    """Construct the deepcell backbone with five convolutional units
        input_tensor (tensor): Input tensor to specify input size
        n_filters (int, optional): Defaults to 32. Number of filters for convolutionaal layers
    Returns:
        (backbone_names, backbone_features): List of backbone layers, list of backbone names
    """


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Build out backbone
    c1 = featurenet_3D_block(img_input, n_filters)  # 1/2 64x64
    c2 = featurenet_3D_block(c1, n_filters)  # 1/4 32x32
    c3 = featurenet_3D_block(c2, n_filters)  # 1/8 16x16
    c4 = featurenet_3D_block(c3, n_filters)  # 1/16 8x8
    c5 = featurenet_3D_block(c4, n_filters)  # 1/32 4x4

    backbone_features = [c1, c2, c3, c4, c5]
    backbone_names = ['C1', 'C2', 'C3', 'C4', 'C5']
    output_dict = {}
    for name, feature in zip(backbone_names, backbone_features):
        output_dict[name] = feature

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=backbone_features)
    return model, output_dict


def get_backbone(backbone, input_tensor, use_imagenet=False, return_dict=True, **kwargs):
    """Retrieve backbones - helper function for the construction of feature pyramid networks
        backbone: Name of the backbone to be retrieved. Options include featurenets, resnets
            densenets, mobilenets, and nasnets
        input_tensor: The tensor to be used as the input for the backbone. Should have channel
            dimension of size 3
        use_imagenet: Defaults to False. Whether to load pre-trained weights for the backbone
        return_dict: Defaults to True. Whether to return a dictionary of backbone layers,
            e.g. {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}. If false, the whole model
            is returned instead
        kwargs: Keyword dictionary for backbone constructions.
            Relevant keys include 'include_top', 'weights' (should be set to None),
            'input_shape', and 'pooling'

    """
    _backbone = str(backbone).lower()

    featurenet_backbones = ['featurenet', 'featurenet3d', 'featurenet_3d']
    vgg_backbones = ['vgg16', 'vgg19']
    densenet_backbones = ['densenet121', 'densenet169', 'densenet201']
    mobilenet_backbones = ['mobilenet', 'mobilenetv2', 'mobilenet_v2']
    resnet_backbones = ['resnet50']
    nasnet_backbones = ['nasnet_large', 'nasnet_mobile']

    # TODO: Check and make sure **kwargs is in the right format.
    # 'weights' flag should be None, and 'input_shape' must have size 3 on the channel axis

    if use_imagenet:
        kwargs_with_weights = copy.copy(kwargs)
        kwargs_with_weights['weights'] = 'imagenet'

    if _backbone in featurenet_backbones:
        if use_imagenet:
            raise ValueError('A featurenet backbone that is pre-trained on imagenet does not exist')

        if '3d' in _backbone:
            model, output_dict = featurenet_3D_backbone(input_tensor=input_tensor, **kwargs)
        else:
            model, output_dict = featurenet_backbone(input_tensor=input_tensor, **kwargs)

        if return_dict:
            return output_dict
        else:
            return model

    if _backbone in vgg_backbones:
        if _backbone == 'vgg16':
            model = applications.VGG16(input_tensor=input_tensor, **kwargs)
        else:
            model = applications.VGG19(input_tensor=input_tensor, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone == 'vgg16':
                model_with_weights = applications.VGG16(**kwargs_with_weights)
            else:
                model_with_weights = applications.VGG19(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
        layer_outputs = [model.get_layer(name=layer_name).output for layer_name in layer_names]

        output_dict = {}
        for i, j in enumerate(layer_names):
            output_dict['C' + str(i + 1)] = layer_outputs[i]
        if return_dict:
            return output_dict
        else:
            return model

    elif _backbone in densenet_backbones:
        if _backbone == 'densenet121':
            model = applications.DenseNet121(input_tensor=input_tensor, **kwargs)
            blocks = [6, 12, 24, 16]
        elif _backbone == 'densenet169':
            model = applications.DenseNet169(input_tensor=input_tensor, **kwargs)
            blocks = [6, 12, 32, 32]
        elif _backbone == 'densenet201':
            model = applications.DenseNet201(input_tensor=input_tensor, **kwargs)
            blocks = [6, 12, 48, 32]

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone == 'densenet121':
                model_with_weights = applications.DenseNet121(**kwargs_with_weights)
            elif _backbone == 'densenet169':
                model_with_weights = applications.DenseNet169(**kwargs_with_weights)
            elif _backbone == 'densenet201':
                model_with_weights = applications.DenseNet201(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['conv1/relu'] + ['conv{}_block{}_concat'.format(idx + 2, block_num)
                                        for idx, block_num in enumerate(blocks)]
        layer_outputs = [model.get_layer(name=layer_name).output for layer_name in layer_names]

        output_dict = {}
        for i, j in enumerate(layer_names):
            output_dict['C%' + str(i + 1)] = layer_outputs[i]
        if return_dict:
            return output_dict
        else:
            return model

    elif _backbone in resnet_backbones:
        model = applications.ResNet50(input_tensor=input_tensor, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = applications.ResNet50(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['bn_conv1', 'res2c_branch2c', 'res3d_branch2c', 'res4f_branch2c', 'res5c_branch2c']
        layer_outputs = [model.get_layer(name=layer_name).output for layer_name in layer_names]

        output_dict = {}
        for i, j in enumerate(layer_names):
            output_dict['C' + str(i+1)] = layer_outputs[i]
        if return_dict:
            return output_dict
        else:
            return model

    elif _backbone in mobilenet_backbones:
        alpha = kwargs.get('alpha', 1.0)
        if _backbone.endswith('v2'):
            model = applications.MobileNetV2(alpha=alpha, input_tensor=input_tensor, **kwargs)
            block_ids = (2, 5, 12)
            layer_names = ['expanded_conv_project_BN'] + ['block_%s_add' % i for i in block_ids] + ['block_16_project_BN']
        else:
            model = applications.MobileNet(alpha=alpha, input_tensor=input_tensor, **kwargs)
            block_ids = (1, 3, 5, 11, 13)
            layer_names = ['conv_pw_%s_relu' % i for i in block_ids]

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone.endswith('v2'):
                model_with_weights = applications.MobileNetV2(alpha=alpha, **kwargs_with_weights)
            else:
                model_with_weights = applications.MobileNet(alpha=alpha, **kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_outputs = [model.get_layer(name=layer_name).output for layer_name in layer_names]

        output_dict = {}
        for i, j in enumerate(layer_names):
            output_dict['C' + str(i+1)] = layer_outputs[i]
        if return_dict:
            return output_dict
        else:
            return model

    elif _backbone in nasnet_backbones:
        if _backbone.endswith('large'):
            model = applications.NASNetLarge(input_tensor=input_tensor, **kwargs)
            block_ids = [5, 12, 18]
        else:
            model = applications.NASNetMobile(input_tensor=input_tensor, **kwargs)
            block_ids = [3, 8, 12]

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone.endswith('large'):
                model_with_weights = applications.NASNetLarge(**kwargs_with_weights)
            else:
                model_with_weights = applications.NASNetMobile(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['stem_bn1', 'reduction_concat_stem_1']
        layer_names += ['normal_concat_%s' % i for i in block_ids]
        layer_outputs = [model.get_layer(name=layer_name).output for layer_name in layer_names]

        output_dict = {}
        for i, j in enumerate(layer_names):
            output_dict['C' + str(i+1)] = layer_outputs[i]
        if return_dict:
            return output_dict
        else:
            return model

    else:
        backbones = list(featurenet_backbones + densenet_backbones +
                         resnet_backbones + vgg_backbones + nasnet_backbones)
        raise ValueError('Invalid value for `backbone`. Must be one of: %s' %
                         ', '.join(backbones))
