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
from tensorflow.python.keras.layers import Input, Concatenate, Add
from tensorflow.python.keras.layers import Activation, BatchNormalization, Softmax
from tensorflow.python.keras.layers import UpSampling2D, UpSampling3D

from deepcell.layers import TensorProduct
from deepcell.layers import ImageNormalization2D, Location2D
from deepcell.layers import UpsampleLike
# from deepcell.model_zoo.fpn import __create_semantic_head
from deepcell.model_zoo.fpn import __create_pyramid_features
from deepcell.utils.backbone_utils import get_backbone
from deepcell.utils.misc_utils import get_sorted_keys


def semantic_upsample(x, n_upsample, n_filters=64, ndim=2):
    """
    Performs iterative rounds of 2x upsampling and
    convolutions with a 3x3 filter to remove aliasing effects

    Args:
        x (tensor): The input tensor to be upsampled
        n_upsample (int): The number of 2x upsamplings
        n_filters (int, optional): Defaults to 64. The number of filters for
            the 3x3 convolution
        ndim (int): The spatial dimensions of the input data.
            Default is 2, but it also works with 3

    Returns:
        tensor: The upsampled tensor
    """
    acceptable_ndims = [2, 3]
    if ndim not in acceptable_ndims:
        raise ValueError('Only 2 and 3 dimensional networks are supported')

    conv = Conv2D if ndim == 2 else Conv3D
    conv_kernel = (3, 3) if ndim == 2 else (1, 3, 3)
    upsampling = UpSampling2D if ndim == 2 else UpSampling3D
    size = (2, 2) if ndim == 2 else (1, 2, 2)
    if n_upsample > 0:
        for i in range(n_upsample):
            x = conv(n_filters, conv_kernel, strides=1,
                     padding='same', data_format='channels_last')(x)
            x = upsampling(size=size)(x)
    else:
        x = conv(n_filters, conv_kernel, strides=1,
                 padding='same', data_format='channels_last')(x)
    return x


def semantic_prediction(semantic_names,
                        semantic_features,
                        target_level=0,
                        input_target=None,
                        n_filters=64,
                        n_dense=64,
                        ndim=2,
                        n_classes=3,
                        semantic_id=0,
                        include_top=True):
    """Creates the prediction head from a list of semantic features

    Args:
        semantic_names (list): A list of the names of the semantic feature layers
        semantic_features (list): A list of semantic features
            NOTE: The semantic_names and semantic features should be in decreasing order
            e.g. [Q6, Q5, Q4, ...]
        target_level (int, optional): Defaults to 0. The level we need to reach.
            Performs 2x upsampling until we're at the target level
        input_target (tensor, optional): Defaults to None. Tensor with the input image.
        n_dense (int, optional): Defaults to 256. The number of filters for dense layers.
        n_classes (int, optional): Defaults to 3.  The number of classes to be predicted.
        semantic_id (int): Defaults to 0. An number to name the final layer. Allows for multiple
            semantic heads.
    Returns:
        tensor: The softmax prediction for the semantic segmentation head

    Raises:
        ValueError: ndim is not 2 or 3
    """

    if n_classes == 1:
        include_top = False

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
    x = TensorProduct(n_classes)(x)

    if include_top:
        x = TensorProduct(n_classes)(x)
        x = Softmax(axis=channel_axis, name='semantic_{}'.format(semantic_id))(x)
    else:
        x = TensorProduct(n_classes)(x)
        x = Activation('relu', name='semantic_{}'.format(semantic_id))(x)

    return x


def __create_semantic_head(pyramid_dict,
                           n_classes=3,
                           n_filters=64,
                           n_dense=128,
                           semantic_id=0,
                           ndim=2,
                           include_top=False,
                           target_level=2,
                           **kwargs):
    """Creates a semantic head from a feature pyramid network.

    Args:
        pyramid_dict: dict of pyramid names and features
        n_classes (int): Defaults to 3.  The number of classes to be predicted
        n_filters (int): Defaults to 64. The number of convolutional filters.
        n_dense (int): Defaults to 128. Number of dense filters.
        semantic_id (int): Defaults to 0.
        ndim (int): Defaults to 2, 3d supported.
        include_top (bool): Defaults to False.
        target_level (int, optional): Defaults to 2. The level we need to reach.
            Performs 2x upsampling until we're at the target level

    Returns:
        keras.layers.Layer: The semantic segmentation head
    """
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

    semantic_sum = pyramid_features[-1]
    semantic_names = pyramid_names[-1]

    # Final upsampling
    min_level = int(re.findall(r'\d+', semantic_names[-1])[0])
    n_upsample = min_level
    x = semantic_upsample(semantic_sum, n_upsample, ndim=ndim)

    # First tensor product
    x = TensorProduct(n_dense)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    # Apply tensor product and softmax layer
    x = TensorProduct(n_classes)(x)

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
                name='panopticnet',
                **kwargs):
    """Constructs a mrcnn model using a backbone from keras-applications.

    Args:
        backbone (str): Name of backbone to use.
        input_shape (tuple): The shape of the input data.
        backbone_levels (list): The backbone levels to be used.
            to create the feature pyramid. Defaults to ['C3', 'C4', 'C5'].
        pyramid_levels (list): Pyramid levels to use. Defaults to ['P3','P4','P5','P6','P7']
        create_pyramid_features (function): Function to get the pyramid
            features from the backbone.
        create_semantic_head (function): Function to get to build a
            semantic head submodel.
        frames_per_batch (int): Defaults to 1.
        temporal_mode: Defaults to None.
        num_semantic_heads (int): Defaults to 1.
        num_semantic_classes (list): Defaults to [3].
        norm_method (str): ImageNormalization mode to use. Defaults to 'whole_image'
        location (bool): Whether to include location data.
        use_imagenet (bool): Whether to load imagenet-based pretrained weights.
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

    Returns:
        tensorflow.keras.Model: Panoptic model with a backbone.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if inputs is None:
        if frames_per_batch > 1:
            if channel_axis == 1:
                input_shape_with_time = tuple(
                    [input_shape[0], frames_per_batch] + list(input_shape)[1:])
            else:
                input_shape_with_time = tuple(
                    [frames_per_batch] + list(input_shape))
            inputs = Input(shape=input_shape_with_time)
        else:
            inputs = Input(shape=input_shape)

    # force the channel size for backbone input to be `required_channels`
    if norm_method is None:
        norm = inputs
    else:
        if frames_per_batch > 1:
            norm = TimeDistributed(ImageNormalization2D(norm_method=norm_method))(inputs)
        else:
            norm = ImageNormalization2D(norm_method=norm_method)(inputs)

    if location:
        if frames_per_batch > 1:
            # TODO: TimeDistributed is incompatible with channels_first
            loc = TimeDistributed(Location2D(in_shape=input_shape))(norm)
        else:
            loc = Location2D(in_shape=input_shape)(norm)
        concat = Concatenate(axis=channel_axis)([norm, loc])
    else:
        concat = norm

    if frames_per_batch > 1:
        fixed_inputs = TimeDistributed(TensorProduct(required_channels))(concat)
    else:
        fixed_inputs = TensorProduct(required_channels)(concat)

    # force the input shape
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
    pyramid_dict = create_pyramid_features(backbone_dict_reduced, ndim=ndim)

    features = [pyramid_dict[key] for key in pyramid_levels]

    if frames_per_batch > 1:
        if temporal_mode in ['conv', 'lstm', 'gru']:
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
