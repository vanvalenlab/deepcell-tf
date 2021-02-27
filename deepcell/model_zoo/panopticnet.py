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
"""Feature pyramid network utility functions"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import re

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import TimeDistributed, ConvLSTM2D
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization

from deepcell.layers import ConvGRU2D
from deepcell.layers import ImageNormalization2D, Location2D
from deepcell.model_zoo.fpn import __create_pyramid_features
from deepcell.model_zoo.fpn import __create_semantic_head
from deepcell.utils.backbone_utils import get_backbone


def __merge_temporal_features(feature, mode='conv', feature_size=256,
                              frames_per_batch=1):
    """Merges feature with its temporal residual through addition.

    Input feature (x) --> Temporal convolution* --> Residual feature (x')
    *Type of temporal convolution specified by ``mode``.

    Output: ``y = x + x'``

    Args:
        feature (tensorflow.keras.Layer): Input layer
        mode (str): Mode of temporal convolution. One of
            ``{'conv','lstm','gru', None}``.
        feature_size (int): Length of convolutional kernel
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.

    Raises:
        ValueError: ``mode`` not 'conv', 'lstm', 'gru' or ``None``

    Returns:
        tensorflow.keras.Layer: Input feature merged with its residual
        from a temporal convolution. If mode is ``None``,
        the output is exactly the input.
    """
    # Check inputs to mode
    acceptable_modes = {'conv', 'lstm', 'gru', None}
    if mode is not None:
        mode = str(mode).lower()
        if mode not in acceptable_modes:
            raise ValueError('Mode {} not supported. Please choose '
                             'from {}.'.format(mode, str(acceptable_modes)))

    f_name = str(feature.name)[:2]

    if mode == 'conv':
        x = Conv3D(feature_size,
                   (frames_per_batch, 3, 3),
                   strides=(1, 1, 1),
                   padding='same',
                   name='conv3D_mtf_{}'.format(f_name),
                   )(feature)
        x = BatchNormalization(axis=-1, name='bnorm_mtf_{}'.format(f_name))(x)
        x = Activation('relu', name='acti_mtf_{}'.format(f_name))(x)
    elif mode == 'lstm':
        x = ConvLSTM2D(feature_size,
                       (3, 3),
                       padding='same',
                       activation='relu',
                       return_sequences=True,
                       name='convLSTM_mtf_{}'.format(f_name))(feature)
    elif mode == 'gru':
        x = ConvGRU2D(feature_size,
                      (3, 3),
                      padding='same',
                      activation='relu',
                      return_sequences=True,
                      name='convGRU_mtf_{}'.format(f_name))(feature)
    else:
        x = feature

    temporal_feature = x

    return temporal_feature


def PanopticNet(backbone,
                input_shape,
                inputs=None,
                backbone_levels=['C3', 'C4', 'C5'],
                pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                create_pyramid_features=__create_pyramid_features,
                create_semantic_head=__create_semantic_head,
                frames_per_batch=1,
                temporal_mode=None,
                num_semantic_classes=[3],
                required_channels=3,
                norm_method=None,
                pooling=None,
                location=True,
                use_imagenet=True,
                lite=False,
                upsample_type='upsampling2d',
                interpolation='bilinear',
                name='panopticnet',
                z_axis_convolutions=False,
                **kwargs):
    """Constructs a Mask-RCNN model using a backbone from
    ``keras-applications`` with optional semantic segmentation transforms.

    Args:
        backbone (str): Name of backbone to use.
        input_shape (tuple): The shape of the input data.
        backbone_levels (list): The backbone levels to be used.
            to create the feature pyramid.
        pyramid_levels (list): Pyramid levels to use.
        create_pyramid_features (function): Function to get the pyramid
            features from the backbone.
        create_semantic_head (function): Function to build a semantic head
            submodel.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        temporal_mode: Mode of temporal convolution. Choose from
            ``{'conv','lstm','gru', None}``.
        num_semantic_classes (list or dict): Number of semantic classes
            for each semantic head. If a ``dict``, keys will be used as
            head names and values will be the number of classes.
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        location (bool): Whether to include a
            :mod:`deepcell.layers.location.Location2D` layer.
        use_imagenet (bool): Whether to load imagenet-based pretrained weights.
        lite (bool): Whether to use a depthwise conv in the feature pyramid
            rather than regular conv.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.
        pooling (str): optional pooling mode for feature extraction
            when ``include_top`` is ``False``.

            - None means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
            - 'avg' means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
            - 'max' means that global max pooling will
              be applied.

        z_axis_convolutions (bool): Whether or not to do convolutions on
            3D data across the z axis.
        required_channels (int): The required number of channels of the
            backbone.  3 is the default for all current backbones.
        kwargs (dict): Other standard inputs for ``retinanet_mask``.

    Raises:
        ValueError: ``temporal_mode`` not 'conv', 'lstm', 'gru'  or ``None``

    Returns:
        tensorflow.keras.Model: Panoptic model with a backbone.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    conv = Conv3D if frames_per_batch > 1 else Conv2D
    conv_kernel = (1, 1, 1) if frames_per_batch > 1 else (1, 1)

    # Check input to __merge_temporal_features
    acceptable_modes = {'conv', 'lstm', 'gru', None}
    if temporal_mode is not None:
        temporal_mode = str(temporal_mode).lower()
        if temporal_mode not in acceptable_modes:
            raise ValueError('temporal_mode {} not supported. Please choose '
                             'from {}.'.format(temporal_mode, acceptable_modes))

    # TODO only works for 2D: do we check for 3D as well?
    # What are the requirements for 3D data?
    img_shape = input_shape[1:] if channel_axis == 1 else input_shape[:-1]
    if img_shape[0] != img_shape[1]:
        raise ValueError('Input data must be square, got dimensions {}'.format(
            img_shape))

    if not math.log(img_shape[0], 2).is_integer():
        raise ValueError('Input data dimensions must be a power of 2, '
                         'got {}'.format(img_shape[0]))

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode "{}" not supported. '
                         'Choose from {}.'.format(
                             interpolation, list(acceptable_interpolation)))

    if inputs is None:
        if frames_per_batch > 1:
            if channel_axis == 1:
                input_shape_with_time = tuple(
                    [input_shape[0], frames_per_batch] + list(input_shape)[1:])
            else:
                input_shape_with_time = tuple(
                    [frames_per_batch] + list(input_shape))
            inputs = Input(shape=input_shape_with_time, name='input_0')
        else:
            inputs = Input(shape=input_shape, name='input_0')

    # Normalize input images
    if norm_method is None:
        norm = inputs
    else:
        if frames_per_batch > 1:
            norm = TimeDistributed(ImageNormalization2D(
                norm_method=norm_method, name='norm'), name='td_norm')(inputs)
        else:
            norm = ImageNormalization2D(norm_method=norm_method,
                                        name='norm')(inputs)

    # Add location layer
    if location:
        if frames_per_batch > 1:
            # TODO: TimeDistributed is incompatible with channels_first
            loc = TimeDistributed(Location2D(name='location'),
                                  name='td_location')(norm)
        else:
            loc = Location2D(name='location')(norm)
        concat = Concatenate(axis=channel_axis,
                             name='concatenate_location')([norm, loc])
    else:
        concat = norm

    # Force the channel size for backbone input to be `required_channels`
    fixed_inputs = conv(required_channels, conv_kernel, strides=1,
                        padding='same', name='conv_channels')(concat)

    # Force the input shape
    axis = 0 if K.image_data_format() == 'channels_first' else -1
    fixed_input_shape = list(input_shape)
    fixed_input_shape[axis] = required_channels
    fixed_input_shape = tuple(fixed_input_shape)

    model_kwargs = {
        'include_top': False,
        'weights': None,
        'input_shape': fixed_input_shape,
        'pooling': pooling
    }

    _, backbone_dict = get_backbone(backbone, fixed_inputs,
                                    use_imagenet=use_imagenet,
                                    frames_per_batch=frames_per_batch,
                                    return_dict=True,
                                    **model_kwargs)

    backbone_dict_reduced = {k: backbone_dict[k] for k in backbone_dict
                             if k in backbone_levels}

    ndim = 2 if frames_per_batch == 1 else 3

    pyramid_dict = create_pyramid_features(backbone_dict_reduced,
                                           ndim=ndim,
                                           lite=lite,
                                           interpolation=interpolation,
                                           upsample_type=upsample_type,
                                           z_axis_convolutions=z_axis_convolutions)

    features = [pyramid_dict[key] for key in pyramid_levels]

    if frames_per_batch > 1:
        temporal_features = [__merge_temporal_features(f, mode=temporal_mode,
                                                       frames_per_batch=frames_per_batch)

                             for f in features]
        for f, k in zip(temporal_features, pyramid_levels):
            pyramid_dict[k] = f

    semantic_levels = [int(re.findall(r'\d+', k)[0]) for k in pyramid_dict]
    target_level = min(semantic_levels)

    semantic_head_list = []
    if not isinstance(num_semantic_classes, dict):
        num_semantic_classes = {
            k: v for k, v in enumerate(num_semantic_classes)
        }

    for k, v in num_semantic_classes.items():
        semantic_head_list.append(create_semantic_head(
            pyramid_dict, n_classes=v,
            input_target=inputs, target_level=target_level,
            semantic_id=k, ndim=ndim, upsample_type=upsample_type,
            interpolation=interpolation, **kwargs))

    outputs = semantic_head_list

    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model
