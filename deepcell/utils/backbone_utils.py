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
"""Functions for creating model backbones"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy

from tensorflow.keras import backend as K
from tensorflow.keras import applications
from tensorflow.keras.backend import is_keras_tensor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPool2D, MaxPool3D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import get_source_inputs


def featurenet_block(x, n_filters):
    """Add a set of layers that make up one unit of the featurenet backbone

    Args:
        x (tensorflow.keras.Layer): Keras layer object to pass to
            backbone unit
        n_filters (int): Number of filters to use for convolutional layers

    Returns:
        tensorflow.keras.Layer: Keras layer object
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
    x = MaxPool2D(pool_size=(2, 2), padding='same', data_format=df)(x)

    return x


def featurenet_3D_block(x, n_filters):
    """Add a set of layers that make up one unit of the featurenet 3D backbone

    Args:
        x (tensorflow.keras.Layer): Keras layer object to pass to
            backbone unit
        n_filters (int): Number of filters to use for convolutional layers

    Returns:
        tensorflow.keras.Layer: Keras layer object
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
        n_filters (int): Number of filters for convolutional layers

    Returns:
        tuple: List of backbone layers, list of backbone names
    """
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not is_keras_tensor(input_tensor):
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
    else:
        if not is_keras_tensor(input_tensor):
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


def get_backbone(backbone, input_tensor=None, input_shape=None,
                 use_imagenet=False, return_dict=True,
                 frames_per_batch=1, **kwargs):
    """Retrieve backbones for the construction of feature pyramid networks.

    Args:
        backbone (str): Name of the backbone to be retrieved.
        input_tensor (tensor): The input tensor for the backbone.
            Should have channel dimension of size 3
        use_imagenet (bool): Load pre-trained weights for the backbone
        return_dict (bool): Whether to return a dictionary of backbone layers,
            e.g. ``{'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'C5': C5}``.
            If false, the whole model is returned instead
        kwargs (dict): Keyword dictionary for backbone constructions.
            Relevant keys include ``'include_top'``,
            ``'weights'`` (should be ``None``),
            ``'input_shape'``, and ``'pooling'``.

    Returns:
        tensorflow.keras.Model: An instantiated backbone

    Raises:
        ValueError: bad backbone name
        ValueError: featurenet backbone with pre-trained imagenet
    """
    _backbone = str(backbone).lower()

    featurenet_backbones = {
        'featurenet': featurenet_backbone,
        'featurenet3d': featurenet_3D_backbone,
        'featurenet_3d': featurenet_3D_backbone
    }
    vgg_backbones = {
        'vgg16': applications.vgg16.VGG16,
        'vgg19': applications.vgg19.VGG19,
    }
    densenet_backbones = {
        'densenet121': applications.densenet.DenseNet121,
        'densenet169': applications.densenet.DenseNet169,
        'densenet201': applications.densenet.DenseNet201,
    }
    mobilenet_backbones = {
        'mobilenet': applications.mobilenet.MobileNet,
        'mobilenetv2': applications.mobilenet_v2.MobileNetV2,
        'mobilenet_v2': applications.mobilenet_v2.MobileNetV2
    }
    resnet_backbones = {
        'resnet50': applications.resnet.ResNet50,
        'resnet101': applications.resnet.ResNet101,
        'resnet152': applications.resnet.ResNet152,
    }
    resnet_v2_backbones = {
        'resnet50v2': applications.resnet_v2.ResNet50V2,
        'resnet101v2': applications.resnet_v2.ResNet101V2,
        'resnet152v2': applications.resnet_v2.ResNet152V2,
    }
    # resnext_backbones = {
    #     'resnext50': applications.resnext.ResNeXt50,
    #     'resnext101': applications.resnext.ResNeXt101,
    # }
    nasnet_backbones = {
        'nasnet_large': applications.nasnet.NASNetLarge,
        'nasnet_mobile': applications.nasnet.NASNetMobile,
    }
    efficientnet_backbones = {
        'efficientnetb0': applications.efficientnet.EfficientNetB0,
        'efficientnetb1': applications.efficientnet.EfficientNetB1,
        'efficientnetb2': applications.efficientnet.EfficientNetB2,
        'efficientnetb3': applications.efficientnet.EfficientNetB3,
        'efficientnetb4': applications.efficientnet.EfficientNetB4,
        'efficientnetb5': applications.efficientnet.EfficientNetB5,
        'efficientnetb6': applications.efficientnet.EfficientNetB6,
        'efficientnetb7': applications.efficientnet.EfficientNetB7,
    }

    # TODO: Check and make sure **kwargs is in the right format.
    # 'weights' flag should be None, and 'input_shape' must have size 3 on the channel axis
    if frames_per_batch == 1:
        if input_tensor is not None:
            img_input = input_tensor
        else:
            if input_shape:
                img_input = Input(shape=input_shape)
            else:
                img_input = Input(shape=(None, None, 3))
    else:
        # using 3D data but a 2D backbone.
        # TODO: why ignore input_tensor
        if input_shape:
            img_input = Input(shape=input_shape)
        else:
            img_input = Input(shape=(None, None, 3))

    if use_imagenet:
        kwargs_with_weights = copy.copy(kwargs)
        kwargs_with_weights['weights'] = 'imagenet'
    else:
        kwargs['weights'] = None

    if _backbone in featurenet_backbones:
        if use_imagenet:
            raise ValueError('A featurenet backbone that is pre-trained on '
                             'imagenet does not exist')

        model_cls = featurenet_backbones[_backbone]
        model, output_dict = model_cls(input_tensor=img_input, **kwargs)

        layer_outputs = [output_dict['C1'], output_dict['C2'], output_dict['C3'],
                         output_dict['C4'], output_dict['C5']]

    elif _backbone in vgg_backbones:
        model_cls = vgg_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in densenet_backbones:
        model_cls = densenet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)
        if _backbone == 'densenet121':
            blocks = [6, 12, 24, 16]
        elif _backbone == 'densenet169':
            blocks = [6, 12, 32, 32]
        elif _backbone == 'densenet201':
            blocks = [6, 12, 48, 32]

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['conv1/relu'] + ['conv{}_block{}_concat'.format(idx + 2, block_num)
                                        for idx, block_num in enumerate(blocks)]
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in resnet_backbones:
        model_cls = resnet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
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
        model_cls = resnet_v2_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
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

    # elif _backbone in resnext_backbones:
    #     model_cls = resnext_backbones[_backbone]
    #     model = model_cls(input_tensor=img_input, **kwargs)
    #
    #     # Set the weights of the model if requested
    #     if use_imagenet:
    #         model_with_weights = model_cls(**kwargs_with_weights)
    #         model_with_weights.save_weights('model_weights.h5')
    #         model.load_weights('model_weights.h5', by_name=True)
    #
    #     if _backbone == 'resnext50':
    #         layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
    #                        'conv4_block6_out', 'conv5_block3_out']
    #     elif _backbone == 'resnext101':
    #         layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out',
    #                        'conv4_block23_out', 'conv5_block3_out']
    #
    #     layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in mobilenet_backbones:
        model_cls = mobilenet_backbones[_backbone]
        alpha = kwargs.pop('alpha', 1.0)
        model = model_cls(alpha=alpha, input_tensor=img_input, **kwargs)
        if _backbone.endswith('v2'):
            block_ids = (2, 5, 12)
            layer_names = ['expanded_conv_project_BN'] + \
                          ['block_%s_add' % i for i in block_ids] + \
                          ['block_16_project_BN']
        else:
            block_ids = (1, 3, 5, 11, 13)
            layer_names = ['conv_pw_%s_relu' % i for i in block_ids]

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(alpha=alpha, **kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in nasnet_backbones:
        model_cls = nasnet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)
        if _backbone.endswith('large'):
            block_ids = [5, 12, 18]
        else:
            block_ids = [3, 8, 12]

        # Set the weights of the model if requested
        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['stem_bn1', 'reduction_concat_stem_1']
        layer_names.extend(['normal_concat_%s' % i for i in block_ids])
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    elif _backbone in efficientnet_backbones:
        model_cls = efficientnet_backbones[_backbone]
        model = model_cls(input_tensor=img_input, **kwargs)

        if use_imagenet:
            model_with_weights = model_cls(**kwargs_with_weights)
            model_with_weights.save_weights('model_weights.h5')
            model.load_weights('model_weights.h5', by_name=True)

        layer_names = ['block2a_expand_activation', 'block3a_expand_activation',
                       'block4a_expand_activation', 'block6a_expand_activation',
                       'top_activation']
        layer_outputs = [model.get_layer(name=ln).output for ln in layer_names]

    else:
        join = lambda x: [v for y in x for v in list(y.keys())]
        backbones = join([featurenet_backbones, densenet_backbones,
                          resnet_backbones, resnet_v2_backbones,
                          vgg_backbones, nasnet_backbones,
                          mobilenet_backbones, efficientnet_backbones])
        raise ValueError('Invalid value for `backbone`. Must be one of: %s' %
                         ', '.join(backbones))

    if frames_per_batch > 1:

        time_distributed_outputs = []
        for i, out in enumerate(layer_outputs):
            td_name = 'td_{}'.format(i)
            model_name = 'model_{}'.format(i)
            time_distributed_outputs.append(
                TimeDistributed(Model(model.input, out, name=model_name),
                                name=td_name)(input_tensor))

        if time_distributed_outputs:
            layer_outputs = time_distributed_outputs

    output_dict = {'C{}'.format(i + 1): j for i, j in enumerate(layer_outputs)}
    return (model, output_dict) if return_dict else model
