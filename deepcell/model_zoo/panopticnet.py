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
"""Feature pyramid network utility functions"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Conv3D
from tensorflow.python.keras.layers import TimeDistributed, ConvLSTM2D
from tensorflow.python.keras.layers import Input, Concatenate
from tensorflow.python.keras.layers import Activation, BatchNormalization, Softmax
from tensorflow.python.keras.layers import UpSampling2D, UpSampling3D

from deepcell.layers import ConvGRU2D
from deepcell.layers import ImageNormalization2D, Location2D
from deepcell.model_zoo.fpn import __create_pyramid_features
from deepcell.utils.backbone_utils import get_backbone
from deepcell.utils.misc_utils import get_sorted_keys


def __merge_temporal_features(feature, mode='conv', feature_size=256,
                              frames_per_batch=1):
    """ Merges feature with its temporal residual through addition.
    Input feature (x) --> Temporal convolution* --> Residual feature (x')
    *Type of temporal convolution specified by "mode" argument
    Output: y = x + x'

    Args:
        feature: Input layer
        mode (str, optional): Mode of temporal convolution. Choose from
            {'conv','lstm','gru', None}. Defaults to 'conv'.
        feature_size (int, optional): Defaults to 256.
        frames_per_batch (int, optional): Defaults to 1.

    Raises:
        ValueError: Mode not 'conv', 'lstm', 'gru' or None

    Returns:
        Input feature merged with its residual from a temporal convolution.
            If mode=None, the output is exactly the input.
    """
    # Check inputs to mode
    acceptable_modes = {'conv', 'lstm', 'gru', None}
    if mode is not None:
        mode = str(mode).lower()
        if mode not in acceptable_modes:
            raise ValueError('Mode {} not supported. Please choose '
                             'from {}.'.format(mode, str(acceptable_modes)))

    if mode == 'conv':
        x = Conv3D(feature_size,
                   (frames_per_batch, 3, 3),
                   strides=(1, 1, 1),
                   padding='same',
                   )(feature)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    elif mode == 'lstm':
        x = ConvLSTM2D(feature_size,
                       (3, 3),
                       padding='same',
                       activation='relu',
                       return_sequences=True)(feature)
    elif mode == 'gru':
        x = ConvGRU2D(feature_size,
                      (3, 3),
                      padding='same',
                      activation='relu',
                      return_sequences=True)(feature)
    else:
        x = feature

    temporal_feature = x

    return temporal_feature


def semantic_upsample(x, n_upsample, n_filters=64, ndim=2,
                      semantic_id=0, interpolation='bilinear'):
    """Performs iterative rounds of 2x upsampling and
    convolutions with a 3x3 filter to remove aliasing effects

    Args:
        x (tensor): The input tensor to be upsampled
        n_upsample (int): The number of 2x upsamplings
        n_filters (int, optional): Defaults to 64. The number of filters for
            the 3x3 convolution
        ndim (int): The spatial dimensions of the input data.
            Default is 2, but it also works with 3
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ['bilinear', 'nearest']. Defaults to bilinear.

    Raises:
        ValueError: ndim is not 2 or 3

    Returns:
        tensor: The upsampled tensor
    """
    # Check input to ndims
    acceptable_ndims = [2, 3]
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode not supported. Choose from '
                         '["bilinear", "nearest"]')

    conv = Conv2D if ndim == 2 else Conv3D
    conv_kernel = (3, 3) if ndim == 2 else (1, 3, 3)
    upsampling = UpSampling2D if ndim == 2 else UpSampling3D
    size = (2, 2) if ndim == 2 else (1, 2, 2)
    if n_upsample > 0:
        for i in range(n_upsample):
            x = conv(n_filters, conv_kernel, strides=1,
                     padding='same', data_format='channels_last',
                     name='conv_{}_semantic_upsample_{}'.format(i, semantic_id))(x)
            x = upsampling(size=size,
                           name='upsampling_{}_semantic_upsample_{}'.format(i, semantic_id),
                           interpolation=interpolation)(x)
    else:
        x = conv(n_filters, conv_kernel, strides=1,
                 padding='same', data_format='channels_last',
                 name='conv_final_semantic_upsample_{}'.format(semantic_id))(x)
    return x


def __create_semantic_head(pyramid_dict,
                           n_classes=3,
                           n_filters=64,
                           n_dense=128,
                           semantic_id=0,
                           ndim=2,
                           include_top=False,
                           target_level=2,
                           interpolation='bilinear',
                           **kwargs):
    """Creates a semantic head from a feature pyramid network.

    Args:
        pyramid_dict (dict): dict of pyramid names and features
        n_classes (int): Defaults to 3.  The number of classes to be predicted
        n_filters (int): Defaults to 64. The number of convolutional filters.
        n_dense (int): Defaults to 128. Number of dense filters.
        semantic_id (int): Defaults to 0.
        ndim (int): Defaults to 2, 3d supported.
        include_top (bool): Defaults to False.
        target_level (int, optional): Defaults to 2. The level we need to reach.
            Performs 2x upsampling until we're at the target level.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ['bilinear', 'nearest']. Defaults to bilinear.

    Raises:
        ValueError: ndim must be 2 or 3

    Returns:
        keras.layers.Layer: The semantic segmentation head
    """
    # Check input to ndims
    if ndim not in {2, 3}:
        raise(ValueError('ndim must be either 2 or 3. '
                         'Received ndim = {}'.format(ndim)))

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode not supported. Choose from '
                         '["bilinear", "nearest"]')

    conv = Conv2D if ndim == 2 else Conv3D
    conv_kernel = (1,) * ndim

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if n_classes == 1:
        include_top = False

    # Get pyramid names and features into list form
    pyramid_names = get_sorted_keys(pyramid_dict)
    pyramid_features = [pyramid_dict[name] for name in pyramid_names]

    # Reverse pyramid names and features
    pyramid_names.reverse()
    pyramid_features.reverse()

    semantic_feature = pyramid_features[-1]
    semantic_name = pyramid_names[-1]

    # Final upsampling
    min_level = int(re.findall(r'\d+', semantic_name[-1])[0])
    n_upsample = min_level
    x = semantic_upsample(semantic_feature, n_upsample, ndim=ndim,
                          interpolation=interpolation, semantic_id=semantic_id)

    # Apply conv in place of previous tensor product
    x = conv(n_dense, conv_kernel, strides=1,
             padding='same', data_format='channels_last',
             name='conv_0_semantic_{}'.format(semantic_id))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu', name='relu_0_semantic_{}'.format(semantic_id))(x)

    # Apply conv and softmax layer
    x = conv(n_classes, conv_kernel, strides=1,
             padding='same', data_format='channels_last',
             name='conv_1_semantic_{}'.format(semantic_id))(x)

    if include_top:
        x = Softmax(axis=channel_axis, name='semantic_{}'.format(semantic_id))(x)
    else:
        x = Activation('relu', name='semantic_{}'.format(semantic_id))(x)

    return x


def PanopticNet(backbone,
                input_shape,
                inputs=None,
                backbone_levels=['C3', 'C4', 'C5'],
                pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                create_pyramid_features=__create_pyramid_features,
                create_semantic_head=__create_semantic_head,
                frames_per_batch=1,
                temporal_mode=None,
                num_semantic_heads=1,
                num_semantic_classes=[3],
                required_channels=3,
                norm_method='whole_image',
                pooling=None,
                location=True,
                use_imagenet=True,
                lite=False,
                interpolation='bilinear',
                name='panopticnet',
                **kwargs):
    """Constructs a mrcnn model using a backbone from keras-applications.

    Args:
        backbone (str): Name of backbone to use.
        input_shape (tuple): The shape of the input data.
        backbone_levels (list): The backbone levels to be used.
            to create the feature pyramid. Defaults to ['C3', 'C4', 'C5'].
        pyramid_levels (list): Pyramid levels to use. Defaults to
            ['P3','P4','P5','P6','P7']
        create_pyramid_features (function): Function to get the pyramid
            features from the backbone.
        create_semantic_head (function): Function to get to build a
            semantic head submodel.
        frames_per_batch (int): Defaults to 1.
        temporal_mode: Mode of temporal convolution. Choose from
            {'conv','lstm','gru', None}. Defaults to None.
        num_semantic_heads (int): Defaults to 1.
        num_semantic_classes (list): Defaults to [3].
        norm_method (str): ImageNormalization mode to use.
            Defaults to 'whole_image'.
        location (bool): Whether to include location data. Defaults to True
        use_imagenet (bool): Whether to load imagenet-based pretrained weights.
        lite (bool): Whether to use a depthwise conv in the feature pyramid
            rather than regular conv. Defaults to False.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ['bilinear', 'nearest']. Defaults to bilinear.
        pooling (str): optional pooling mode for feature extraction
            when include_top is False.

            - None means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - 'avg' means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - 'max' means that global max pooling will
                be applied.

        required_channels (int): The required number of channels of the
            backbone.  3 is the default for all current backbones.
        kwargs (dict): Other standard inputs for retinanet_mask.

    Raises:
        ValueError: temporal_mode not 'conv', 'lstm', 'gru'  or None

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
            raise ValueError('Mode {} not supported. Please choose '
                             'from {}.'.format(temporal_mode,
                                               str(acceptable_modes)))

    
    # TODO only works for 2D: do we check for 3D as well? What are the requirements for 3D data?
    img_shape = input_shape[1:] if channel_axis == 1 else input_shape[:-1]
    if img_shape[0] != img_shape[1]:
        raise ValueError('Input data must be square, got dimensions {}'.format(img_shape))

    if img_shape[0] not in [2 ** x for x in range(5, 12)]:
        raise ValueError('Input data dimensions must be a power of 2, got {}'.format(img_shape[0]))

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode not supported. Choose from '
                         '["bilinear", "nearest"]')

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
                norm_method=norm_method, name='norm'))(inputs)
        else:
            norm = ImageNormalization2D(norm_method=norm_method,
                                        name='norm')(inputs)

    # Add location layer
    if location:
        if frames_per_batch > 1:
            # TODO: TimeDistributed is incompatible with channels_first
            loc = TimeDistributed(Location2D(in_shape=input_shape,
                                             name='location'))(norm)
        else:
            loc = Location2D(in_shape=input_shape, name='location')(norm)
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
                                    return_dict=True, **model_kwargs)

    backbone_dict_reduced = {k: backbone_dict[k] for k in backbone_dict
                             if k in backbone_levels}
    ndim = 2 if frames_per_batch == 1 else 3
    pyramid_dict = create_pyramid_features(backbone_dict_reduced,
                                           ndim=ndim,
                                           lite=lite,
                                           upsample_type='upsampling2d')

    features = [pyramid_dict[key] for key in pyramid_levels]

    if frames_per_batch > 1:
        temporal_features = [__merge_temporal_features(
            feature, mode=temporal_mode) for feature in features]
        for f, k in zip(temporal_features, pyramid_dict.keys()):
            pyramid_dict[k] = f

    semantic_levels = [int(re.findall(r'\d+', k)[0]) for k in pyramid_dict]
    target_level = min(semantic_levels)

    semantic_head_list = []
    for i in range(num_semantic_heads):
        semantic_head_list.append(create_semantic_head(
            pyramid_dict, n_classes=num_semantic_classes[i],
            input_target=inputs, target_level=target_level,
            semantic_id=i, ndim=ndim, **kwargs))

    outputs = semantic_head_list

    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model
