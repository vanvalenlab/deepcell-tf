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

import keras_applications as applications

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, Conv3D, BatchNormalization
from tensorflow.python.keras.layers import Activation, MaxPool2D, MaxPool3D
from tensorflow.python.keras import backend as K

try:
    from tensorflow.python.keras.backend import is_keras_tensor
except ImportError:
    from tensorflow.python.keras._impl.keras.backend import is_keras_tensor

try:
    from tensorflow.python.keras.utils.data_utils import get_file
except ImportError:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file


try:
    from tensorflow.python.keras.utils.layer_utils import get_source_inputs
except ImportError:
    try:
        from tensorflow.python.keras.engine.network import get_source_inputs
    except ImportError:  # tf1.8 uses the _impl directory
        from tensorflow.python.keras._impl.keras.engine.network import get_source_inputs


def featurenet_block(x, n_filters):
    """Add a set of layers that make up one unit of the featurenet backbone

    Args:
        x (tensorflow.keras.layers.Layer): Keras layer object to pass to
            backbone unit
        n_filters (int): Number of filters to use for convolutional layers

    Returns:
        tensorflow.keras.layers.Layer: Keras layer object
    """
    df = K.image_data_format()
    # conv set 1
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', data_format=df)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # conv set 2
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same', data_format=df)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # Final max pooling stage
    x = MaxPool2D(pool_size=(2, 2), data_format=df)(x)

    return x


def featurenet_3D_block(x, n_filters):
    """Add a set of layers that make up one unit of the featurenet 3D backbone

    Args:
        x (tensorflow.keras.layers.Layer): Keras layer object to pass to
            backbone unit
        n_filters (int): Number of filters to use for convolutional layers

    Returns:
        tensorflow.keras.layers.Layer: Keras layer object
    """
    df = K.image_data_format()
    # conv set 1
    x = Conv3D(n_filters, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format=df)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # conv set 2
    x = Conv3D(n_filters, (3, 3, 3), strides=(1, 1, 1), padding='same', data_format=df)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    # Final max pooling stage
    x = MaxPool3D(pool_size=(2, 2, 2), data_format=df)(x)

    return x


def featurenet_backbone(input_tensor=None, input_shape=None,
                        n_filters=32, **kwargs):
    """Construct the deepcell backbone with five convolutional units

    Args:
        input_tensor (tensor): Input tensor to specify input size
        n_filters (int): Defaults to 32. Number of filters for
            convolutional layers

    Returns:
        tuple: List of backbone layers, list of backbone names
    """
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    elif not K.is_keras_tensor(input_tensor):
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
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=backbone_features)
    return model, output_dict


def featurenet_3D_backbone(input_tensor=None, input_shape=None,
                           n_filters=32, **kwargs):
    """Construct the deepcell backbone with five convolutional units

    Args:
        input_tensor (tensor): Input tensor to specify input size
        n_filters (int): Number of filters for convolutional layers

    Returns:
        tuple: List of backbone layers, list of backbone names
    """
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    elif not K.is_keras_tensor(input_tensor):
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
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=backbone_features)
    return model, output_dict


def get_backbone(backbone, input_tensor, use_imagenet=False, return_dict=True, **kwargs):
    """Retrieve backbones - helper function for the construction of feature pyramid networks

    Args:
        backbone (str): Name of the backbone to be retrieved.
        input_tensor (tensor): The input tensor for the backbone.
            Should have channel dimension of size 3
        use_imagenet (bool): Load pre-trained weights for the backbone
        return_dict (bool): Whether to return a dictionary of backbone layers,
            e.g. {'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}.
            If false, the whole model is returned instead
        kwargs (dict): Keyword dictionary for backbone constructions.
            Relevant keys include 'include_top', 'weights' (should be set to None),
            'input_shape', and 'pooling'

    Returns:
        tensorflow.keras.Model: An instantiated backbone

    Raises:
        ValueError: bad backbone name
        ValueError: featurenet backbone with pre-trained imagenet
    """
    _backbone = str(backbone).lower()

    # set up general Utils class to deal with different tf versions
    class Utils(object):  # pylint: disable=useless-object-inheritance
        pass

    utils = Utils()
    utils.get_file = get_file
    utils.get_source_inputs = get_source_inputs

    K.is_keras_tensor = is_keras_tensor

    kwargs['backend'] = K
    kwargs['layers'] = tf.keras.layers
    kwargs['models'] = tf.keras.models
    kwargs['utils'] = utils

    featurenet_backbones = ['featurenet', 'featurenet3d', 'featurenet_3d']
    vgg_backbones = ['vgg16', 'vgg19']
    densenet_backbones = ['densenet121', 'densenet169', 'densenet201']
    mobilenet_backbones = ['mobilenet', 'mobilenetv2', 'mobilenet_v2']
    resnet_backbones = ['resnet50', 'resnet101', 'resnet152']
    resnet_v2_backbones = ['resnet50v2', 'resnet101v2', 'resnet152v2']
    resnext_backbones = ['resnext50', 'resnext101']
    nasnet_backbones = ['nasnet_large', 'nasnet_mobile']

    # TODO: Check and make sure **kwargs is in the right format.
    # 'weights' flag should be None, and 'input_shape' must have size 3 on the channel axis

    if use_imagenet:
        kwargs_with_weights = copy.copy(kwargs)
        kwargs_with_weights['weights'] = 'imagenet'
    else:
        kwargs['weights'] = None

    if _backbone in featurenet_backbones:
        if use_imagenet:
            raise ValueError('A featurenet backbone that is pre-trained on '
                             'imagenet does not exist')

        if '3d' in _backbone:
            model, output_dict = featurenet_3D_backbone(input_tensor=input_tensor, **kwargs)
        else:
            model, output_dict = featurenet_backbone(input_tensor=input_tensor, **kwargs)

        layer_outputs = [output_dict['C1'], output_dict['C2'], output_dict['C3'],
                         output_dict['C4'], output_dict['C5']]

    elif _backbone in vgg_backbones:
        if _backbone == 'vgg16':
            model = applications.vgg16.VGG16(input_tensor=input_tensor, **kwargs)
        else:
            model = applications.vgg19.VGG19(input_tensor=input_tensor, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone == 'vgg16':
                model_with_weights = applications.vgg16.VGG16(**kwargs_with_weights)
            else:
                model_with_weights = applications.vgg19.VGG19(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in densenet_backbones:
        if _backbone == 'densenet121':
            model = applications.densenet.DenseNet121(input_tensor=input_tensor, **kwargs)
            blocks = [6, 12, 24, 16]
        elif _backbone == 'densenet169':
            model = applications.densenet.DenseNet169(input_tensor=input_tensor, **kwargs)
            blocks = [6, 12, 32, 32]
        elif _backbone == 'densenet201':
            model = applications.densenet.DenseNet201(input_tensor=input_tensor, **kwargs)
            blocks = [6, 12, 48, 32]

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone == 'densenet121':
                model_with_weights = applications.densenet.DenseNet121(**kwargs_with_weights)
            elif _backbone == 'densenet169':
                model_with_weights = applications.densenet.DenseNet169(**kwargs_with_weights)
            elif _backbone == 'densenet201':
                model_with_weights = applications.densenet.DenseNet201(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['conv1/relu'] + ['conv{}_block{}_concat'.format(idx + 2, block_num)
                                        for idx, block_num in enumerate(blocks)]
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in resnet_backbones:
        if _backbone == 'resnet50':
            model = applications.resnet.ResNet50(input_tensor=input_tensor, **kwargs)
        elif _backbone == 'resnet101':
            model = applications.resnet.ResNet101(input_tensor=input_tensor, **kwargs)
        elif _backbone == 'resnet152':
            model = applications.resnet.ResNet152(input_tensor=input_tensor, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone == 'resnet50':
                model_with_weights = applications.resnet.ResNet50(**kwargs_with_weights)
            elif _backbone == 'resnet101':
                model_with_weights = applications.resnet.ResNet101(**kwargs_with_weights)
            elif _backbone == 'resnet152':
                model_with_weights = applications.resnet.ResNet152(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        if _backbone == 'resnet50':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block6_out', 'conv5_block3_out']
        elif _backbone == 'resnet101':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block23_out', 'conv5_block3_out']
        elif _backbone == 'resnet152':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block8_out',
                           'conv4_block36_out', 'conv5_block3_out']

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in resnet_v2_backbones:
        if _backbone == 'resnet50v2':
            model = applications.resnet_v2.ResNet50V2(input_tensor=input_tensor, **kwargs)
        elif _backbone == 'resnet101v2':
            model = applications.resnet_v2.ResNet101V2(input_tensor=input_tensor, **kwargs)
        elif _backbone == 'resnet152v2':
            model = applications.resnet_v2.ResNet152V2(input_tensor=input_tensor, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone == 'resnet50v2':
                model_with_weights = applications.resnet_v2.ResNet50V2(**kwargs_with_weights)
            elif _backbone == 'resnet101v2':
                model_with_weights = applications.resnet_v2.ResNet101V2(**kwargs_with_weights)
            elif _backbone == 'resnet152v2':
                model_with_weights = applications.resnet_v2.ResNet152V2(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        if _backbone == 'resnet50v2':
            layer_names = ['post_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block6_out', 'conv5_block3_out']
        elif _backbone == 'resnet101v2':
            layer_names = ['post_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block23_out', 'conv5_block3_out']
        elif _backbone == 'resnet152v2':
            layer_names = ['post_relu', 'conv2_block3_out', 'conv3_block8_out',
                           'conv4_block36_out', 'conv5_block3_out']

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in resnext_backbones:
        if _backbone == 'resnext50':
            model = applications.resnext.ResNeXt50(input_tensor=input_tensor, **kwargs)
        elif _backbone == 'resnext101':
            model = applications.resnext.ResNeXt101(input_tensor=input_tensor, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone == 'resnext50':
                model_with_weights = applications.resnext.ResNeXt50(**kwargs_with_weights)
            elif _backbone == 'resnext101':
                model_with_weights = applications.resnext.ResNeXt101(**kwargs_with_weights)

            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        if _backbone == 'resnext50':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block6_out', 'conv5_block3_out']
        elif _backbone == 'resnext101':
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
                           'conv4_block23_out', 'conv5_block3_out']

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in mobilenet_backbones:
        alpha = kwargs.get('alpha', 1.0)
        if _backbone.endswith('v2'):
            model = applications.mobilenet_v2.MobileNetV2(
                alpha=alpha, input_tensor=input_tensor, **kwargs)
            block_ids = (2, 5, 12)
            layer_names = ['expanded_conv_project_BN'] + \
                          ['block_%s_add' % i for i in block_ids] + \
                          ['block_16_project_BN']

        else:
            model = applications.mobilenet.MobileNet(
                alpha=alpha, input_tensor=input_tensor, **kwargs)
            block_ids = (1, 3, 5, 11, 13)
            layer_names = ['conv_pw_%s_relu' % i for i in block_ids]

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone.endswith('v2'):
                model_with_weights = applications.mobilenet_v2.MobileNetV2(
                    alpha=alpha, **kwargs_with_weights)
            else:
                model_with_weights = applications.mobilenet.MobileNet(
                    alpha=alpha, **kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in nasnet_backbones:
        if _backbone.endswith('large'):
            model = applications.nasnet.NASNetLarge(input_tensor=input_tensor, **kwargs)
            block_ids = [5, 12, 18]
        else:
            model = applications.nasnet.NASNetMobile(input_tensor=input_tensor, **kwargs)
            block_ids = [3, 8, 12]

        # Set the weights of the model if requested
        if use_imagenet:
            if _backbone.endswith('large'):
                model_with_weights = applications.nasnet.NASNetLarge(**kwargs_with_weights)
            else:
                model_with_weights = applications.nasnet.NASNetMobile(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['stem_bn1', 'reduction_concat_stem_1']
        layer_names.extend(['normal_concat_%s' % i for i in block_ids])
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    else:
        backbones = list(featurenet_backbones + densenet_backbones +
                         resnet_backbones + resnext_backbones +
                         resnet_v2_backbones + vgg_backbones +
                         nasnet_backbones + mobilenet_backbones)
        raise ValueError('Invalid value for `backbone`. Must be one of: %s' %
                         ', '.join(backbones))

    output_dict = {'C{}'.format(i + 1): j for i, j in enumerate(layer_outputs)}
    return output_dict if return_dict else model
