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
"""Feature pyramid network utility functions"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, DepthwiseConv2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.keras.layers import BatchNormalization

from deepcell.layers import UpsampleLike
from deepcell.utils.misc_utils import get_sorted_keys


def create_pyramid_level(backbone_input,
                         upsamplelike_input=None,
                         addition_input=None,
                         upsample_type='upsamplelike',
                         level=5,
                         ndim=2,
                         lite=False,
                         interpolation='bilinear',
                         feature_size=256,
                         z_axis_convolutions=False):
    """Create a pyramid layer from a particular backbone input layer.

    Args:
        backbone_input (tensorflow.keras.Layer): Backbone layer to use to
            create they pyramid layer.
        upsamplelike_input (tensor): Optional input to use
            as a template for shape to upsample to.
        addition_input (tensorflow.keras.Layer): Optional layer to add to
            pyramid layer after convolution and upsampling.
        upsample_type (str): Choice of upsampling methods
            from ``['upsamplelike','upsampling2d','upsampling3d']``.
        level (int): Level to use in layer names.
        feature_size (int): Number of filters for the convolutional layer.
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        lite (bool): Whether to use depthwise conv instead of regular conv for
            feature pyramid construction
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Returns:
        tuple: Pyramid layer after processing, upsampled pyramid layer

    Raises:
        ValueError: ``ndim`` is not 2 or 3
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``
    """
    # Check input to ndims
    acceptable_ndims = {2, 3}
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    # Check if inputs to ndim and lite are compatible
    if ndim == 3 and lite:
        raise ValueError('lite models are not compatible with 3 dimensional '
                         'networks')

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode "{}" not supported. '
                         'Choose from {}.'.format(
                             interpolation, list(acceptable_interpolation)))

    # Check input to upsample_type
    acceptable_upsample = {'upsamplelike', 'upsampling2d', 'upsampling3d'}
    if upsample_type not in acceptable_upsample:
        raise ValueError('Upsample method "{}" not supported. '
                         'Choose from {}.'.format(
                             upsample_type, list(acceptable_upsample)))

    reduced_name = 'C{}_reduced'.format(level)
    upsample_name = 'P{}_upsampled'.format(level)
    addition_name = 'P{}_merged'.format(level)
    final_name = 'P{}'.format(level)

    # Apply 1x1 conv to backbone layer
    if ndim == 2:
        pyramid = Conv2D(feature_size, (1, 1), strides=(1, 1),
                         padding='same', name=reduced_name)(backbone_input)
    else:
        pyramid = Conv3D(feature_size, (1, 1, 1), strides=(1, 1, 1),
                         padding='same', name=reduced_name)(backbone_input)

    # Add and then 3x3 conv
    if addition_input is not None:
        pyramid = Add(name=addition_name)([pyramid, addition_input])

    # Upsample pyramid input
    if upsamplelike_input is not None and upsample_type == 'upsamplelike':
        pyramid_upsample = UpsampleLike(name=upsample_name)(
            [pyramid, upsamplelike_input])
    elif upsample_type == 'upsamplelike':
        pyramid_upsample = None
    else:
        upsampling = UpSampling2D if ndim == 2 else UpSampling3D
        size = (2, 2) if ndim == 2 else (1, 2, 2)
        upsampling_kwargs = {
            'size': size,
            'name': upsample_name,
            'interpolation': interpolation
        }
        if ndim > 2:
            del upsampling_kwargs['interpolation']
        pyramid_upsample = upsampling(**upsampling_kwargs)(pyramid)

    if ndim == 2:
        if lite:
            pyramid_final = DepthwiseConv2D((3, 3), strides=(1, 1),
                                            padding='same',
                                            name=final_name)(pyramid)
        else:
            pyramid_final = Conv2D(feature_size, (3, 3), strides=(1, 1),
                                   padding='same', name=final_name)(pyramid)
    else:
        z = 3 if z_axis_convolutions else 1
        pyramid_final = Conv3D(feature_size, (z, 3, 3), strides=(1, 1, 1),
                               padding='same', name=final_name)(pyramid)

    return pyramid_final, pyramid_upsample


def __create_pyramid_features(backbone_dict,
                              ndim=2,
                              feature_size=256,
                              include_final_layers=True,
                              lite=False,
                              upsample_type='upsamplelike',
                              interpolation='bilinear',
                              z_axis_convolutions=False):
    """Creates the FPN layers on top of the backbone features.

    Args:
        backbone_dict (dictionary): A dictionary of the backbone layers, with
            the names as keys, e.g. ``{'C0': C0, 'C1': C1, 'C2': C2, ...}``
        feature_size (int): The feature size to use for
            the resulting feature levels.
        include_final_layers (bool): Add two coarser pyramid levels
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        lite (bool): Whether to use depthwise conv instead of regular conv for
            feature pyramid construction
        upsample_type (str): Choice of upsampling methods
            from ``['upsamplelike','upsamling2d','upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Returns:
        dict: The feature pyramid names and levels,
        e.g. ``{'P3': P3, 'P4': P4, ...}``
        Each backbone layer gets a pyramid level, and two additional levels
        are added, e.g. ``[C3, C4, C5]`` --> ``[P3, P4, P5, P6, P7]``

    Raises:
        ValueError: ``ndim`` is not 2 or 3
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``
    """
    # Check input to ndims
    acceptable_ndims = [2, 3]
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    # Check if inputs to ndim and lite are compatible
    if ndim == 3 and lite:
        raise ValueError('lite models are not compatible with 3 dimensional '
                         'networks')

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode "{}" not supported. '
                         'Choose from {}.'.format(
                             interpolation, list(acceptable_interpolation)))

    # Check input to upsample_type
    acceptable_upsample = {'upsamplelike', 'upsampling2d', 'upsampling3d'}
    if upsample_type not in acceptable_upsample:
        raise ValueError('Upsample method "{}" not supported. '
                         'Choose from {}.'.format(
                             upsample_type, list(acceptable_upsample)))

    # Get names of the backbone levels and place in ascending order
    backbone_names = get_sorted_keys(backbone_dict)
    backbone_features = [backbone_dict[name] for name in backbone_names]

    pyramid_names = []
    pyramid_finals = []
    pyramid_upsamples = []

    # Reverse lists
    backbone_names.reverse()
    backbone_features.reverse()

    for i, N in enumerate(backbone_names):
        level = int(re.findall(r'\d+', N)[0])
        pyramid_names.append('P{}'.format(level))

        backbone_input = backbone_features[i]

        # Don't add for the bottom of the pyramid
        if i == 0:
            if len(backbone_features) > 1:
                upsamplelike_input = backbone_features[i + 1]
            else:
                upsamplelike_input = None
            addition_input = None

        # Don't upsample for the top of the pyramid
        elif i == len(backbone_names) - 1:
            upsamplelike_input = None
            addition_input = pyramid_upsamples[-1]

        # Otherwise, add and upsample
        else:
            upsamplelike_input = backbone_features[i + 1]
            addition_input = pyramid_upsamples[-1]

        pf, pu = create_pyramid_level(backbone_input,
                                      upsamplelike_input=upsamplelike_input,
                                      addition_input=addition_input,
                                      upsample_type=upsample_type,
                                      level=level,
                                      ndim=ndim,
                                      lite=lite,
                                      interpolation=interpolation,
                                      z_axis_convolutions=z_axis_convolutions)
        pyramid_finals.append(pf)
        pyramid_upsamples.append(pu)

    # Add the final two pyramid layers
    if include_final_layers:
        # "Second to last pyramid layer is obtained via a
        # 3x3 stride-2 conv on the coarsest backbone"
        N = backbone_names[0]
        F = backbone_features[0]
        level = int(re.findall(r'\d+', N)[0]) + 1
        P_minus_2_name = 'P{}'.format(level)

        if ndim == 2:
            P_minus_2 = Conv2D(feature_size, kernel_size=(3, 3),
                               strides=(2, 2), padding='same',
                               name=P_minus_2_name)(F)
        else:
            P_minus_2 = Conv3D(feature_size, kernel_size=(1, 3, 3),
                               strides=(1, 2, 2), padding='same',
                               name=P_minus_2_name)(F)

        pyramid_names.insert(0, P_minus_2_name)
        pyramid_finals.insert(0, P_minus_2)

        # "Last pyramid layer is computed by applying ReLU
        # followed by a 3x3 stride-2 conv on second to last layer"
        level = int(re.findall(r'\d+', N)[0]) + 2
        P_minus_1_name = 'P{}'.format(level)
        P_minus_1 = Activation('relu', name='{}_relu'.format(N))(P_minus_2)

        if ndim == 2:
            P_minus_1 = Conv2D(feature_size, kernel_size=(3, 3),
                               strides=(2, 2), padding='same',
                               name=P_minus_1_name)(P_minus_1)
        else:
            P_minus_1 = Conv3D(feature_size, kernel_size=(1, 3, 3),
                               strides=(1, 2, 2), padding='same',
                               name=P_minus_1_name)(P_minus_1)

        pyramid_names.insert(0, P_minus_1_name)
        pyramid_finals.insert(0, P_minus_1)

    pyramid_dict = dict(zip(pyramid_names, pyramid_finals))

    return pyramid_dict


def semantic_upsample(x,
                      n_upsample,
                      target=None,
                      n_filters=64,
                      ndim=2,
                      semantic_id=0,
                      upsample_type='upsamplelike',
                      interpolation='bilinear'):
    """Performs iterative rounds of 2x upsampling and
    convolutions with a 3x3 filter to remove aliasing effects.

    Args:
        x (tensor): The input tensor to be upsampled.
        n_upsample (int): The number of 2x upsamplings.
        target (tensor): An optional tensor with the target shape.
        n_filters (int): The number of filters for
            the 3x3 convolution.
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        semantic_id (int): ID of the semantic head.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Raises:
        ValueError: ``ndim`` is not 2 or 3.
        ValueError: ``interpolation`` not in ``['bilinear', 'nearest']``.
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``.
        ValueError: ``target`` is ``None`` and
            ``upsample_type`` is ``'upsamplelike'``

    Returns:
        tensor: The upsampled tensor.
    """
    # Check input to ndims
    acceptable_ndims = [2, 3]
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode "{}" not supported. '
                         'Choose from {}.'.format(
                             interpolation, list(acceptable_interpolation)))

    # Check input to upsample_type
    acceptable_upsample = {'upsamplelike', 'upsampling2d', 'upsampling3d'}
    if upsample_type not in acceptable_upsample:
        raise ValueError('Upsample method "{}" not supported. '
                         'Choose from {}.'.format(
                             upsample_type, list(acceptable_upsample)))

    # Check that there is a target if upsamplelike is used
    if upsample_type == 'upsamplelike' and target is None:
        raise ValueError('upsamplelike requires a target.')

    conv = Conv2D if ndim == 2 else Conv3D
    conv_kernel = (3, 3) if ndim == 2 else (1, 3, 3)
    upsampling = UpSampling2D if ndim == 2 else UpSampling3D
    size = (2, 2) if ndim == 2 else (1, 2, 2)

    if n_upsample > 0:
        for i in range(n_upsample):
            x = conv(n_filters, conv_kernel, strides=1, padding='same',
                     name='conv_{}_semantic_upsample_{}'.format(
                         i, semantic_id))(x)

            # Define kwargs for upsampling layer
            upsample_name = 'upsampling_{}_semantic_upsample_{}'.format(
                i, semantic_id)

            if upsample_type == 'upsamplelike':
                if i == n_upsample - 1 and target is not None:
                    x = UpsampleLike(name=upsample_name)([x, target])
            else:
                upsampling_kwargs = {
                    'size': size,
                    'name': upsample_name,
                    'interpolation': interpolation
                }

                if ndim > 2:
                    del upsampling_kwargs['interpolation']
                x = upsampling(**upsampling_kwargs)(x)
    else:
        x = conv(n_filters, conv_kernel, strides=1, padding='same',
                 name='conv_final_semantic_upsample_{}'.format(semantic_id))(x)

        if upsample_type == 'upsamplelike' and target is not None:
            upsample_name = 'upsampling_{}_semanticupsample_{}'.format(
                0, semantic_id)
            x = UpsampleLike(name=upsample_name)([x, target])

    return x


def __create_semantic_head(pyramid_dict,
                           input_target=None,
                           n_classes=3,
                           n_filters=128,
                           n_dense=128,
                           semantic_id=0,
                           ndim=2,
                           include_top=False,
                           target_level=2,
                           upsample_type='upsamplelike',
                           interpolation='bilinear',
                           **kwargs):
    """Creates a semantic head from a feature pyramid network.

    Args:
        pyramid_dict (dict): Dictionary of pyramid names and features.
        input_target (tensor): Optional tensor with the input image.
        n_classes (int): The number of classes to be predicted.
        n_filters (int): The number of convolutional filters.
        n_dense (int): Number of dense filters.
        semantic_id (int): ID of the semantic head.
        ndim (int): The spatial dimensions of the input data.
            Must be either 2 or 3.
        include_top (bool): Whether to include the final layer of the model
        target_level (int): The level we need to reach. Performs
            2x upsampling until we're at the target level.
        upsample_type (str): Choice of upsampling layer to use from
            ``['upsamplelike', 'upsampling2d', 'upsampling3d']``.
        interpolation (str): Choice of interpolation mode for upsampling
            layers from ``['bilinear', 'nearest']``.

    Raises:
        ValueError: ``ndim`` must be 2 or 3
        ValueError: ``interpolation`` not in ``['bilinear', 'nearest']``
        ValueError: ``upsample_type`` not in
            ``['upsamplelike','upsampling2d', 'upsampling3d']``

    Returns:
        tensorflow.keras.Layer: The semantic segmentation head
    """
    # Check input to ndims
    if ndim not in {2, 3}:
        raise ValueError('ndim must be either 2 or 3. '
                         'Received ndim = {}'.format(ndim))

    # Check input to interpolation
    acceptable_interpolation = {'bilinear', 'nearest'}
    if interpolation not in acceptable_interpolation:
        raise ValueError('Interpolation mode "{}" not supported. '
                         'Choose from {}.'.format(
                             interpolation, list(acceptable_interpolation)))

    # Check input to upsample_type
    acceptable_upsample = {'upsamplelike', 'upsampling2d', 'upsampling3d'}
    if upsample_type not in acceptable_upsample:
        raise ValueError('Upsample method "{}" not supported. '
                         'Choose from {}.'.format(
                             upsample_type, list(acceptable_upsample)))

    # Check that there is an input_target if upsamplelike is used
    if upsample_type == 'upsamplelike' and input_target is None:
        raise ValueError('upsamplelike requires an input_target.')

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

    # Previous method of building feature pyramids
    # semantic_features, semantic_names = [], []
    # for N, P in zip(pyramid_names, pyramid_features):
    #     # Get level and determine how much to upsample
    #     level = int(re.findall(r'\d+', N)[0])
    #
    #     n_upsample = level - target_level
    #     target = semantic_features[-1] if len(semantic_features) > 0 else None
    #
    #     # Use semantic upsample to get semantic map
    #     semantic_features.append(semantic_upsample(
    #         P, n_upsample, n_filters=n_filters, target=target, ndim=ndim,
    #         upsample_type=upsample_type, interpolation=interpolation,
    #         semantic_id=semantic_id))
    #     semantic_names.append('Q{}'.format(level))

    # Add all the semantic features
    # semantic_sum = semantic_features[0]
    # for semantic_feature in semantic_features[1:]:
    #     semantic_sum = Add()([semantic_sum, semantic_feature])

    # TODO: bad name but using the same name more clearly indicates
    # how to integrate the previous version
    semantic_sum = pyramid_features[-1]

    # Final upsampling
    # min_level = int(re.findall(r'\d+', pyramid_names[-1])[0])
    # n_upsample = min_level - target_level
    n_upsample = target_level
    x = semantic_upsample(semantic_sum, n_upsample,
                          # n_filters=n_filters,  # TODO: uncomment and retrain
                          target=input_target, ndim=ndim,
                          upsample_type=upsample_type, semantic_id=semantic_id,
                          interpolation=interpolation)

    # Apply conv in place of previous tensor product
    x = conv(n_dense, conv_kernel, strides=1, padding='same',
             name='conv_0_semantic_{}'.format(semantic_id))(x)
    x = BatchNormalization(axis=channel_axis,
                           name='batch_normalization_0_semantic_{}'.format(semantic_id))(x)
    x = Activation('relu', name='relu_0_semantic_{}'.format(semantic_id))(x)

    # Apply conv and softmax layer
    x = conv(n_classes, conv_kernel, strides=1,
             padding='same', name='conv_1_semantic_{}'.format(semantic_id))(x)

    if include_top:
        x = Softmax(axis=channel_axis,
                    dtype=K.floatx(),
                    name='semantic_{}'.format(semantic_id))(x)
    else:
        x = Activation('relu',
                       dtype=K.floatx(),
                       name='semantic_{}'.format(semantic_id))(x)

    return x
