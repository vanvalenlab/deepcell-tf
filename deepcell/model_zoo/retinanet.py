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
"""RetinaNet models adapted from https://github.com/fizyr/keras-retinanet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, TimeDistributed
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import Permute, Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.initializers import RandomNormal

from deepcell.initializers import PriorProbability
from deepcell.layers import TensorProduct
from deepcell.layers import FilterDetections
from deepcell.layers import ImageNormalization2D, Location2D
from deepcell.layers import Anchors, RegressBoxes, ClipBoxes
from deepcell.utils.retinanet_anchor_utils import AnchorParameters
from deepcell.model_zoo.fpn import __create_semantic_head
from deepcell.model_zoo.fpn import __create_pyramid_features
from deepcell.utils.backbone_utils import get_backbone


def default_classification_model(num_classes,
                                 num_anchors,
                                 pyramid_feature_size=256,
                                 prior_probability=0.01,
                                 classification_feature_size=256,
                                 frames_per_batch=1,
                                 name='classification_submodel'):
    """Creates the default regression submodel.

    Args:
        num_classes (int): Number of classes to predict a score
            for at each feature level.
        num_anchors (int): Number of anchors to predict classification
            scores for at each feature level.
        pyramid_feature_size (int): The number of filters to expect from the
            feature pyramid levels.
        prior_probability (float): the prior probability
        classification_feature_size (int): The number of filters to use in the
            layers in the classification submodel.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        name (str): The name of the submodel.

    Returns:
        tensorflow.keras.Model: A model that predicts classes for each anchor.
    """
    time_distributed = frames_per_batch > 1

    options = {
        'kernel_size': (3, 3, 3) if time_distributed else 3,
        'strides': 1,
        'padding': 'same',
    }

    shape = [None] * (4 if time_distributed else 3)
    if K.image_data_format() == 'channels_first':
        shape[0] = pyramid_feature_size
    else:
        shape[-1] = pyramid_feature_size
    inputs = Input(shape=shape)
    outputs = inputs
    conv = Conv3D if time_distributed else Conv2D
    for i in range(4):
        outputs = conv(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = conv(
        filters=num_classes * num_anchors,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if K.image_data_format() == 'channels_first':
        rank = 4 if time_distributed else 3
        perm = tuple(list(range(2, rank + 1)) + [1])
        outputs = Permute(perm, name='pyramid_classification_permute')(outputs)

    new_shape = (frames_per_batch, -1, num_classes)
    if not time_distributed:
        new_shape = new_shape[1:]

    outputs = Reshape(new_shape, name='pyramid_classification_reshape')(outputs)
    outputs = Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values,
                             num_anchors,
                             pyramid_feature_size=256,
                             regression_feature_size=256,
                             frames_per_batch=1,
                             name='regression_submodel'):
    """Creates the default regression submodel.

    Args:
        num_values (int): Number of values to regress.
        num_anchors (int): Number of anchors to regress for each feature level.
        pyramid_feature_size (int): The number of filters to expect from the
            feature pyramid levels.
        regression_feature_size (int): The number of filters to use in the layers
            in the regression submodel.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        name (str): The name of the submodel.

    Returns:
        tensorflow.keras.Model: A model that predicts regression values
        for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    time_distributed = frames_per_batch > 1

    options = {
        'kernel_size': (3, 3, 3) if time_distributed else 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    shape = [None] * (4 if time_distributed else 3)
    if K.image_data_format() == 'channels_first':
        shape[0] = pyramid_feature_size
    else:
        shape[-1] = pyramid_feature_size
    inputs = Input(shape=shape)
    outputs = inputs
    conv = Conv3D if time_distributed else Conv2D
    for i in range(4):
        outputs = conv(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = conv(num_anchors * num_values,
                   name='pyramid_regression', **options)(outputs)

    if K.image_data_format() == 'channels_first':
        rank = 4 if time_distributed else 3
        perm = tuple(list(range(2, rank + 1)) + [1])
        outputs = Permute(perm, name='pyramid_regression_permute')(outputs)

    new_shape = (frames_per_batch, -1, num_values)
    if not time_distributed:
        new_shape = new_shape[1:]

    outputs = Reshape(new_shape, name='pyramid_regression_reshape')(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


def default_submodels(num_classes, num_anchors, frames_per_batch=1):
    """Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel
    and a classification submodel.

    Args:
        num_classes (int): Number of classes to use.
        num_anchors (int): Number of base anchors.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.

    Returns:
        list: A list of tuples, where the first element is the name of the
        submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(
            4, num_anchors, frames_per_batch=frames_per_batch)),
        ('classification', default_classification_model(
            num_classes, num_anchors, frames_per_batch=frames_per_batch))
    ]


def __build_model_pyramid(name, model, features):
    """Applies a single submodel to each FPN level.

    Args:
        name (str): Name of the submodel.
        model (str): The submodel to evaluate.
        features (list): The FPN features.

    Returns:
        tensor: The response from the submodel on the FPN features.
    """
    if len(features) == 1:
        identity = Activation('linear', name=name)
        return identity(model(features[0]))
    else:
        concat = Concatenate(axis=-2, name=name)
        return concat([model(f) for f in features])


def __build_pyramid(models, features):
    """Applies all submodels to each FPN level.

    Args:
        models (list): List of submodels to run on each pyramid level
            (by default only regression, classifcation).
        features (list): The FPN features.

    Returns:
        list: A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features, frames_per_batch=1):
    """Builds anchors for the shape of the features from FPN.

    Args:
        anchor_parameters (AnchorParameters): Parameters that determine how
            anchors are generated.
        features (list): The FPN features.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.

    Returns:
        tensor: The anchors for the FPN features.
        The shape is: ``(batch_size, num_anchors, 4)``
    """

    if len(features) == 1:
        if frames_per_batch > 1:
            anchors = TimeDistributed(Anchors(
                size=anchor_parameters.sizes[0],
                stride=anchor_parameters.strides[0],
                ratios=anchor_parameters.ratios,
                scales=anchor_parameters.scales,
                name='anchors'))(features[0])
        else:
            anchors = Anchors(
                size=anchor_parameters.sizes[0],
                stride=anchor_parameters.strides[0],
                ratios=anchor_parameters.ratios,
                scales=anchor_parameters.scales,
                name='anchors')(features[0])
        return anchors
    else:
        if frames_per_batch > 1:
            anchors = [
                TimeDistributed(Anchors(
                    size=anchor_parameters.sizes[i],
                    stride=anchor_parameters.strides[i],
                    ratios=anchor_parameters.ratios,
                    scales=anchor_parameters.scales,
                    name='anchors_{}'.format(i)
                ))(f) for i, f in enumerate(features)
            ]
        else:
            anchors = [
                Anchors(
                    size=anchor_parameters.sizes[i],
                    stride=anchor_parameters.strides[i],
                    ratios=anchor_parameters.ratios,
                    scales=anchor_parameters.scales,
                    name='anchors_{}'.format(i)
                )(f) for i, f in enumerate(features)
            ]
        return Concatenate(axis=-2, name='anchors')(anchors)


def retinanet(inputs,
              backbone_dict,
              num_classes,
              backbone_levels=['C3', 'C4', 'C5'],
              pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
              num_anchors=None,
              create_pyramid_features=__create_pyramid_features,
              create_semantic_head=__create_semantic_head,
              panoptic=False,
              num_semantic_heads=1,
              num_semantic_classes=[3],
              submodels=None,
              frames_per_batch=1,
              semantic_only=False,
              name='retinanet'):
    """Construct a ``RetinaNet`` model on top of a backbone.

    This model is the minimum model necessary for training
    (with the unfortunate exception of anchors as output).

    Args:
        inputs (tensor): The inputs to the network.
        backbone_dict (dict): A dictionary with the backbone layers.
        backbone_levels (list): The backbone levels to be used.
            to create the feature pyramid.
        pyramid_levels (list): The pyramid levels to attach regression and
            classification heads.
        num_classes (int): Number of classes to classify.
        num_anchors (int): Number of base anchors.
        create_pyramid_features (function): Function to create pyramid features.
        create_semantic_head (function): Function for creating a semantic head,
            which can be used for panoptic segmentation tasks.
        panoptic (bool): Flag for adding the semantic head for panoptic
            segmentation tasks.
        num_semantic_heads (int): The number of semantic segmentation heads.
        num_semantic_classes (list): The number of classes for the semantic
            segmentation part of panoptic segmentation tasks.
        submodels (list): Submodels to run on each feature map
            (default is regression and classification submodels).
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        name (str): Name of the model.

    Returns:
        tensorflow.keras.Model: A Model which takes an image as input
        and outputs generated anchors and the result from each submodel on
        every pyramid level.

        The order of the outputs is as defined in submodels:

        .. code-block:: python

            [
                regression, classification, other[0], other[1], ...
            ]

    """
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors,
                                      frames_per_batch=frames_per_batch)

    if not isinstance(num_semantic_classes, list):
        num_semantic_classes = list(num_semantic_classes)

    # compute pyramid features as per https://arxiv.org/abs/1708.02002

    # Use only the desired backbone levels to create the feature pyramid
    backbone_dict_reduced = {k: backbone_dict[k] for k in backbone_dict
                             if k in backbone_levels}
    pyramid_dict = create_pyramid_features(
        backbone_dict_reduced, ndim=3 if frames_per_batch > 1 else 2)

    # for the desired pyramid levels, run available submodels
    features = [pyramid_dict[key] for key in pyramid_levels]
    object_head = __build_pyramid(submodels, features)

    if panoptic:
        semantic_levels = [int(re.findall(r'\d+', k)[0]) for k in pyramid_dict]
        target_level = min(semantic_levels)

        semantic_head_list = []
        for i in range(num_semantic_heads):
            semantic_head_list.append(create_semantic_head(
                pyramid_dict, n_classes=num_semantic_classes[i],
                input_target=inputs, target_level=target_level,
                semantic_id=i, ndim=3 if frames_per_batch > 1 else 2))

        outputs = object_head + semantic_head_list
    else:
        outputs = object_head

    if semantic_only:
        outputs = semantic_head_list

    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.backbone_levels = backbone_levels
    model.pyramid_levels = pyramid_levels

    return model


def retinanet_bbox(model=None,
                   nms=True,
                   panoptic=False,
                   num_semantic_heads=1,
                   class_specific_filter=True,
                   name='retinanet-bbox',
                   anchor_params=None,
                   max_detections=300,
                   frames_per_batch=1,
                   **kwargs):
    """Construct a RetinaNet model on top of a backbone and adds convenience
    functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers
    to compute boxes within the graph. These layers include applying the
    regression values to the anchors and performing NMS.

    Args:
        model (tensorflow.keras.Model): RetinaNet model to append bbox
            layers to. If None, it will create a RetinaNet model using kwargs.
        nms (bool): Whether to use non-maximum suppression
            for the filtering step.
        panoptic (bool): Flag for adding the semantic head for panoptic
            segmentation tasks.
        num_semantic_heads (int): The number of semantic segmentation heads.
        class_specific_filter (bool): Whether to use class specific filtering
            or filter for the best scoring class only.
        name (str): Name of the model.
        anchor_params (AnchorParameters): Struct containing anchor parameters.
            If None, default values are used.
        max_detections (int): The maximum number of detections allowed.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        kwargs (dict): Additional kwargs to pass to the minimal retinanet model.

    Returns:
        tensorflow.keras.Model: A Model which takes an image as input and
        outputs the detections on the image.

        The order is defined as follows:

        .. code-block:: python

            [
                boxes, scores, labels, other[0], other[1], ...
            ]

    Raises:
        ValueError: the given model does not have a regression or
            classification submodel.
    """
    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(),
                          frames_per_batch=frames_per_batch, **kwargs)
    else:
        names = ('regression', 'classification')
        if not all(output in model.output_names for output in names):
            raise ValueError('Input is not a training model (no `regression` '
                             'and `classification` outputs were found, '
                             'outputs are: {}).'.format(model.output_names))

    # compute the anchors
    features = [model.get_layer(l).output for l in model.pyramid_levels]
    anchors = __build_anchors(anchor_params, features,
                              frames_per_batch=frames_per_batch)

    # we expect anchors, regression. and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default []
    if panoptic:
        # The last output is the panoptic output, which should not be
        # sent to filter detections
        other = model.outputs[2:-num_semantic_heads]
        semantic = model.outputs[-num_semantic_heads:]
    else:
        other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        max_detections=max_detections,
        name='filtered_detections'
    )([boxes, classification] + other)

    # add the semantic head's output if needed
    if panoptic:
        outputs = detections + list(semantic)
    else:
        outputs = detections

    # construct the model
    return Model(inputs=model.inputs, outputs=outputs, name=name)


def RetinaNet(backbone,
              num_classes,
              input_shape,
              inputs=None,
              norm_method='whole_image',
              location=False,
              use_imagenet=False,
              pooling=None,
              required_channels=3,
              frames_per_batch=1,
              **kwargs):
    """Constructs a RetinaNet model using a backbone from
    ``keras-applications``.

    Args:
        backbone (str): Name of backbone to use.
        num_classes (int): Number of classes to classify.
        input_shape (tuple): The shape of the input data.
        inputs (tensor): Optional input tensor, overrides ``input_shape``.
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        location (bool): Whether to include a
            :mod:`deepcell.layers.location.Location2D` layer.
        use_imagenet (bool): Whether to load imagenet-based pretrained weights.
        pooling (str): Pooling mode for feature extraction
            when ``include_top`` is ``False``.

            - None means that the output of the model will be
              the 4D tensor output of the last convolutional layer.
            - 'avg' means that global average pooling will be applied to
              the output of the last convolutional layer, and thus
              the output of the model will be a 2D tensor.
            - 'max' means that global max pooling will be applied.

        required_channels (int): The required number of channels of the
            backbone. 3 is the default for all current backbones.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        kwargs (dict): Other standard inputs for `~retinanet`.

    Returns:
        tensorflow.keras.Model: RetinaNet model with a backbone.
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
            inputs = Input(shape=input_shape_with_time, name='input')
        else:
            inputs = Input(shape=input_shape, name='input')

    if location:
        if frames_per_batch > 1:
            # TODO: TimeDistributed is incompatible with channels_first
            loc = TimeDistributed(Location2D(in_shape=input_shape))(inputs)
        else:
            loc = Location2D(in_shape=input_shape)(inputs)
        concat = Concatenate(axis=channel_axis)([inputs, loc])
    else:
        concat = inputs

    # force the channel size for backbone input to be `required_channels`
    if frames_per_batch > 1:
        norm = TimeDistributed(ImageNormalization2D(norm_method=norm_method))(concat)
        fixed_inputs = TimeDistributed(TensorProduct(required_channels))(norm)
    else:
        norm = ImageNormalization2D(norm_method=norm_method)(concat)
        fixed_inputs = TensorProduct(required_channels)(norm)

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

    # create the full model
    return retinanet(
        inputs=inputs,
        num_classes=num_classes,
        backbone_dict=backbone_dict,
        frames_per_batch=frames_per_batch,
        name='{}_retinanet'.format(backbone),
        **kwargs)
