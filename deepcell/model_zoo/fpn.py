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
from tensorflow.python.keras.layers import Softmax
from tensorflow.python.keras.layers import Input, Add
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.python.keras.layers import BatchNormalization

from deepcell.layers import UpsampleLike
from deepcell.layers import TensorProduct, ImageNormalization2D
from deepcell.utils.backbone_utils import get_backbone
from deepcell.utils.misc_utils import get_sorted_keys


def create_pyramid_level(backbone_input,
                         upsamplelike_input=None,
                         addition_input=None,
                         level=5,
                         ndim=2,
                         feature_size=256):
    """Create a pyramid layer from a particular backbone input layer.

    Args:
        backbone_input (layer): Backbone layer to use to create they pyramid layer
        upsamplelike_input (tensor): Optional input to use
            as a template for shape to upsample to
        addition_input (layer): Optional layer to add to
            pyramid layer after convolution and upsampling.
        level (int): Level to use in layer names, defaults to 5.
        feature_size (int):Number of filters for
            convolutional layer, defaults to 256.
        ndim (int): The spatial dimensions of the input data. Default is 2,
            but it also works with 3

    Returns:
        tuple: Pyramid layer after processing, upsampled pyramid layer

    Raises:
        ValueError: ndim is not 2 or 3
    """

    acceptable_ndims = {2, 3}
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

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

    # Upsample pyramid input
    if upsamplelike_input is not None:
        pyramid_upsample = UpsampleLike(name=upsample_name)(
            [pyramid, upsamplelike_input])
    else:
        pyramid_upsample = None

    # Add and then 3x3 conv
    if addition_input is not None:
        pyramid = Add(name=addition_name)([pyramid, addition_input])

    if ndim == 2:
        pyramid_final = Conv2D(feature_size, (3, 3), strides=(1, 1),
                               padding='same', name=final_name)(pyramid)
    else:
        pyramid_final = Conv3D(feature_size, (1, 3, 3), strides=(1, 1, 1),
                               padding='same', name=final_name)(pyramid)

    return pyramid_final, pyramid_upsample


def __create_pyramid_features(backbone_dict, ndim=2, feature_size=256,
                              include_final_layers=True):
    """Creates the FPN layers on top of the backbone features.

    Args:
        backbone_dict (dictionary): A dictionary of the backbone layers, with
            the names as keys, e.g. {'C0': C0, 'C1': C1, 'C2': C2, ...}
        feature_size (int): Defaults to 256. The feature size to use
            for the resulting feature levels.
        include_final_layers (bool): Add two coarser pyramid levels
        ndim (int): The spatial dimensions of the input data.
            Default is 2, but it also works with 3

    Returns:
        dict: The feature pyramid names and levels,
            e.g. {'P3': P3, 'P4': P4, ...}
            Each backbone layer gets a pyramid level, and two additional levels
            are added, e.g. [C3, C4, C5] --> [P3, P4, P5, P6, P7]

    Raises:
        ValueError: ndim is not 2 or 3
    """

    acceptable_ndims = [2, 3]
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    # Get names of the backbone levels and place in ascending order
    backbone_names = get_sorted_keys(backbone_dict)
    backbone_features = [backbone_dict[name] for name in backbone_names]

    pyramid_names = []
    pyramid_finals = []
    pyramid_upsamples = []

    # Reverse lists
    backbone_names.reverse()
    backbone_features.reverse()

    for i in range(len(backbone_names)):

        N = backbone_names[i]
        level = int(re.findall(r'\d+', N)[0])
        p_name = 'P{}'.format(level)
        pyramid_names.append(p_name)

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
                                      level=level,
                                      ndim=ndim)
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
            P_minus_2 = Conv2D(feature_size, kernel_size=(3, 3), strides=(2, 2),
                               padding='same', name=P_minus_2_name)(F)
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
        P_minus_1 = Activation('relu', name=N + '_relu')(P_minus_2)

        if ndim == 2:
            P_minus_1 = Conv2D(feature_size, kernel_size=(3, 3), strides=(2, 2),
                               padding='same', name=P_minus_1_name)(P_minus_1)
        else:
            P_minus_1 = Conv3D(feature_size, kernel_size=(1, 3, 3),
                               strides=(1, 2, 2), padding='same',
                               name=P_minus_1_name)(P_minus_1)

        pyramid_names.insert(0, P_minus_1_name)
        pyramid_finals.insert(0, P_minus_1)

    pyramid_names.reverse()
    pyramid_finals.reverse()

    # Reverse lists
    backbone_names.reverse()
    backbone_features.reverse()

    pyramid_dict = {}
    for name, feature in zip(pyramid_names, pyramid_finals):
        pyramid_dict[name] = feature

    return pyramid_dict


def semantic_upsample(x, n_upsample, n_filters=64, ndim=2, target=None):
    """
    Performs iterative rounds of 2x upsampling and
    convolutions with a 3x3 filter to remove aliasing effects

    Args:
        x (tensor): The input tensor to be upsampled
        n_upsample (int): The number of 2x upsamplings
        n_filters (int): Defaults to 256. The number of filters for
            the 3x3 convolution
        target (tensor): An optional tensor with the target shape.
            If included, then the final upsampling layer will reshape
            to the target tensor's size
        ndim (int): The spatial dimensions of the input data.
            Default is 2, but it also works with 3

    Returns:
        tensor: The upsampled tensor

    Raises:
        ValueError: ndim is not in {2, 3}.
    """
    acceptable_ndims = [2, 3]
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    conv = Conv2D if ndim == 2 else Conv3D
    upsampling = UpSampling2D if ndim == 2 else UpSampling3D

    for i in range(n_upsample):
        x = conv(n_filters, 3, strides=1,
                 padding='same', data_format='channels_last')(x)

        if i == n_upsample - 1 and target is not None:
            x = UpsampleLike()([x, target])
        else:
            x = upsampling(size=2)(x)

    if n_upsample == 0:
        x = conv(n_filters, 3, strides=1,
                 padding='same', data_format='channels_last')(x)

        if target is not None:
            x = UpsampleLike()([x, target])

    return x


def semantic_prediction(semantic_names,
                        semantic_features,
                        target_level=0,
                        input_target=None,
                        n_filters=64,
                        n_dense=64,
                        ndim=2,
                        n_classes=3,
                        semantic_id=0):
    """Creates the prediction head from a list of semantic features

    Args:
        semantic_names (list): A list of the names of the semantic feature layers
        semantic_features (list): A list of semantic features
            NOTE: The semantic_names and semantic features should be in decreasing order
            e.g. [Q6, Q5, Q4, ...]
        target_level (int): (Optional) The level we need to reach.
            Performs 2x upsampling until we're at the target level.
        input_target (tensor): Optional tensor with the input image.
        n_filters (int): The number of filters for the 3x3 convolution.
        n_dense (int): The number of filters for dense layers.
        ndim (int): The spatial dimensions of the input data.
            Default is 2, but it also works with 3.
        n_classes (int): The number of classes to be predicted.
        semantic_id (int): Defaults to 0. A number to name the final layer.
            Allows for multiple semantic heads.
    Returns:
        tensor: The softmax prediction for the semantic segmentation head

    Raises:
        ValueError: ndim is not 2 or 3
    """

    acceptable_ndims = [2, 3]
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Add all the semantic layers
    semantic_sum = semantic_features[0]
    for semantic_feature in semantic_features[1:]:
        semantic_sum = Add()([semantic_sum, semantic_feature])

    # Final upsampling
    min_level = int(re.findall(r'\d+', semantic_names[-1])[0])
    n_upsample = min_level - target_level
    x = semantic_upsample(semantic_sum, n_upsample,
                          target=input_target, ndim=ndim)

    # First tensor product
    x = TensorProduct(n_dense)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    # Apply tensor product and softmax layer
    if n_classes > 1:
        x = TensorProduct(n_classes)(x)
        x = Softmax(axis=channel_axis, name='semantic_{}'.format(semantic_id))(x)
    elif n_classes == 1:
        x = TensorProduct(n_classes)(x)
        x = Activation('relu', name='semantic_{}'.format(semantic_id))(x)

    return x


def __create_semantic_head(pyramid_dict,
                           input_target=None,
                           target_level=2,
                           n_classes=3,
                           n_filters=128,
                           semantic_id=0,
                           ndim=2):
    """
    Creates a semantic head from a feature pyramid network
    Args:
        pyramid_dict (dict): Pyramid names and features
        input_target (tensor): Optional tensor with the input image.
        target_level (int): Upsampling level.
            Level 1 = 1/2^1 size, Level 2 = 1/2^2 size, Level 3 = 1/2^3 size, etc.
        n_classes (int): The number of classes to be predicted
        n_filters (int): The number of convolutional filters.
        semantic_id (int): A number to name the final layer.
            Allows for multiple semantic heads.
        ndim (int): The spatial dimensions of the input data.
            Default is 2, but it also works with 3.
    Returns:
        keras.layers.Layer: The semantic segmentation head
    """
    # Get pyramid names and features into list form
    pyramid_names = get_sorted_keys(pyramid_dict)
    pyramid_features = [pyramid_dict[name] for name in pyramid_names]

    # Reverse pyramid names and features
    pyramid_names.reverse()
    pyramid_features.reverse()

    semantic_features = []
    semantic_names = []

    for N, P in zip(pyramid_names, pyramid_features):
        # Get level and determine how much to upsample
        level = int(re.findall(r'\d+', N)[0])

        n_upsample = level - target_level
        target = semantic_features[-1] if len(semantic_features) > 0 else None

        # Use semantic upsample to get semantic map
        semantic_features.append(semantic_upsample(
            P, n_upsample, n_filters=n_filters, target=target, ndim=ndim))
        semantic_names.append('Q{}'.format(level))

    # Combine all of the semantic features
    x = semantic_prediction(semantic_names, semantic_features,
                            n_classes=n_classes, input_target=input_target,
                            semantic_id=semantic_id, ndim=ndim)

    return x


def FPNet(backbone,
          input_shape,
          inputs=None,
          norm_method='whole_image',
          use_imagenet=False,
          pooling=None,
          required_channels=3,
          n_classes=3,
          name='fpnet',
          frames_per_batch=1,
          **kwargs):
    """Creates a Feature Pyramid Network with a semantic segmentation head

    Args:
        backbone (str): A name of a supported backbone from [deepcell, resnet50]
        input_shape (tuple): Shape of the input image.
        inputs (keras.Layer): Optional preexisting layers.
        norm_method (str): Normalization method, defaults to 'whole_image'
        use_imagenet (bool): Whether to load imagenet-based pretrained weights.
        pooling (str): Optional pooling mode for feature extraction
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
            backbone. 3 is the default for all current backbones.
        n_classes (int): The number of classes to be predicted
        name (str): Name to use for the model.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.

    Returns:
        tensorflow.keras.models.Model: Feature pyramid network with a semantic
            segmentation head as the output
    """

    if inputs is None:
        inputs = Input(shape=input_shape)

    # force the channel size for backbone input to be required_channels
    norm = ImageNormalization2D(norm_method=norm_method)(inputs)
    fixed_inputs = TensorProduct(required_channels)(norm)

    # force the input shape
    fixed_input_shape = list(input_shape)
    fixed_input_shape[-1] = required_channels
    fixed_input_shape = tuple(fixed_input_shape)

    model_kwargs = {
        'include_top': False,
        'weights': None,
        'input_shape': fixed_input_shape,
        'pooling': pooling
    }

    # Get backbone outputs
    _, backbone_dict = get_backbone(backbone, fixed_inputs,
                                    use_imagenet=use_imagenet,
                                    frames_per_batch=frames_per_batch,
                                    return_dict=True, **model_kwargs)

    # Construct feature pyramid network
    pyramid_dict = __create_pyramid_features(backbone_dict)

    levels = [int(re.findall(r'\d+', k)[0]) for k in pyramid_dict]
    target_level = min(levels)

    x = __create_semantic_head(pyramid_dict, n_classes=n_classes,
                               input_target=inputs, target_level=target_level,
                               ndim=len(input_shape) - 1)

    return Model(inputs=inputs, outputs=x, name=name)
