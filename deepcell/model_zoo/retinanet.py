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
"""RetinaNet models adapted from https://github.com/fizyr/keras-retinanet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Input, Concatenate, Add
from tensorflow.python.keras.layers import Permute, Reshape
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.initializers import RandomNormal

from deepcell.initializers import PriorProbability
from deepcell.layers import TensorProduct
from deepcell.layers import FilterDetections
from deepcell.layers import ImageNormalization2D
from deepcell.layers import Anchors, UpsampleLike, RegressBoxes, ClipBoxes
from deepcell.utils.retinanet_anchor_utils import AnchorParameters, generate_anchor_params
from deepcell.utils.misc_utils import get_backbone
from deepcell.utils.fpn_utils import __create_pyramid_features, __create_semantic_head

def default_classification_model(num_classes,
                                 num_anchors,
                                 pyramid_feature_size=256,
                                 prior_probability=0.01,
                                 classification_feature_size=256,
                                 name='classification_submodel'):
    """Creates the default regression submodel.

    Args:
        num_classes: Number of classes to predict a score for at each feature level.
        num_anchors: Number of anchors to predict classification
            scores for at each feature level.
        pyramid_feature_size: The number of filters to expect from the
            feature pyramid levels.
        classification_feature_size: The number of filters to use in the layers
            in the classification submodel.
        name: The name of the submodel.

    Returns:
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    if K.image_data_format() == 'channels_first':
        inputs = Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if K.image_data_format() == 'channels_first':
        outputs = Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values,
                             num_anchors,
                             pyramid_feature_size=256,
                             regression_feature_size=256,
                             name='regression_submodel'):
    """Creates the default regression submodel.

    Args:
        num_values: Number of values to regress.
        num_anchors: Number of anchors to regress for each feature level.
        pyramid_feature_size: The number of filters to expect from the
            feature pyramid levels.
        regression_feature_size: The number of filters to use in the layers
            in the regression submodel.
        name: The name of the submodel.

    Returns:
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    if K.image_data_format() == 'channels_first':
        inputs = Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if K.image_data_format() == 'channels_first':
        outputs = Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)

def default_submodels(num_classes, num_anchors):
    """Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel
    and a classification submodel.

    Args:
        num_classes: Number of classes to use.
        num_anchors: Number of base anchors.

    Returns:
        A list of tuple, where the first element is the name of the submodel
        and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    """Applies a single submodel to each FPN level.

    Args:
        name: Name of the submodel.
        model: The submodel to evaluate.
        features: The FPN features.

    Returns:
        A tensor containing the response from the submodel on the FPN features.
    """
    concat = Concatenate(axis=1, name=name)
    return concat([model(f) for f in features])


def __build_pyramid(models, features):
    """Applies all submodels to each FPN level.

    Args:
        models: List of sumodels to run on each pyramid level
            (by default only regression, classifcation).
        features: The FPN features.

    Returns:
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]

def __build_anchors(anchor_parameters, features):
    """Builds anchors for the shape of the features from FPN.

    Args:
        anchor_parameters: Parameters that determine how anchors are generated.
        features: The FPN features.

    Returns:
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return Concatenate(axis=1, name='anchors')(anchors)


def retinanet(inputs,
              backbone_dict,
              num_classes,
              backbone_levels=['C3', 'C4', 'C5'],
              pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
              num_anchors=None,
              create_pyramid_features=__create_pyramid_features,
              create_symantic_head=__create_semantic_head,
              panoptic=False,
              num_semantic_classes=3,
              submodels=None,
              name='retinanet'):
    """Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training
    (with the unfortunate exception of anchors as output).

    Args:
        inputs: The inputs to the network.
        backbone_dict: A dictionary with the backbone layers
        backbone_levels: A list with the backbone levels to be used
            to create the feature pyramid. Defaults to ['C3', 'C4', 'C5']
        pyramid_levels: A list of the pyramid levels to attach regression and
            classification heads to. Defaults to ['P3', 'P4', 'P5', 'P6', 'P7']
        num_classes: Number of classes to classify.
        num_anchors: Number of base anchors.
        create_pyramid_features: Function for creating pyramid features.
        create_symantic_head: Function for creating a semantic head, which can 
            be used for panoptic segmentation tasks
        panoptic: Flag for adding the semantic head for panoptic segmentation 
            tasks. Defaults to false.
        num_semantic_classes: The number of classes for the semantic segmentation
            part of panoptic segmentation tasks. Defaults to 3.
        submodels: Submodels to run on each feature map (default is regression
            and classification submodels).
        name: Name of the model.

    Returns:
        A Model which takes an image as input and outputs generated anchors
        and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    # compute pyramid features as per https://arxiv.org/abs/1708.02002

    # Use only the desired backbone levels to create the feature pyramid
    backbone_dict_reduced = {key: value for key, value in backbone_dict.items() 
                                if key in backbone_levels}
    pyramid_dict = create_pyramid_features(backbone_dict_reduced)

    # for the desired pyramid levels, run available submodels
    features = [pyramid_dict[key] for key in pyramid_levels]
    object_head = __build_pyramid(submodels, features)

    if panoptic:
        semantic_levels = [int(re.findall(r'\d+', N)[0]) for N in pyramid_dict.keys()]
        target_level = min(semantic_levels)

        semantic_head = __create_semantic_head(pyramid_dict, n_classes=num_semantic_classes,
                               input_target=inputs, target_level=target_level)

        outputs = object_head + [semantic_head]
    else:
        outputs = object_head

    return Model(inputs=inputs, outputs=outputs, name=name)


def retinanet_bbox(model=None,
                   nms=True,
                   backbone_levels=['C3', 'C4', 'C5'],
                   pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                   panoptic=False,
                   class_specific_filter=True,
                   name='retinanet-bbox',
                   anchor_params=None,
                   **kwargs):
    """Construct a RetinaNet model on top of a backbone and adds convenience
    functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to
    compute boxes within the graph. These layers include applying the regression
    values to the anchors and performing NMS.

    Args:
        model: RetinaNet model to append bbox layers to.
            If None, it will create a RetinaNet model using **kwargs.
        nms: Whether to use non-maximum suppression for the filtering step.
        backbone_levels: Backbone levels to use for constructing retinanet.
        pyramid_levels: Pyramid levels to attach the object detection heads to.
        class_specific_filter: Whether to use class specific filtering or
            filter for the best scoring class only.
        name: Name of the model.
        anchor_params: Struct containing anchor parameters.
            If None, default values are used.
        **kwargs: Additional kwargs to pass to the minimal retinanet model.

    Returns:
        A Model which takes an image as input and
        outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """
    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), 
                            backbone_levels=backbone_levels,
                            pyramid_levels=pyramid_levels, 
                            **kwargs)
    else:
        names = ('regression', 'classification')
        if not all(output in model.output_names for output in names):
            raise ValueError('Input is not a training model (no `regression` '
                             'and `classification` outputs were found, '
                             'outputs are: {}).'.format(model.output_names))

    # compute the anchors
    features = [model.get_layer(level).output for level in pyramid_levels]
    anchors = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    if panoptic:
        # The last output is the panoptic output, which should not be
        # sent to filter detections
        other = model.outputs[2:-1]
    else:
        other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification] + other)

    # add the semantic head's output if needed
    if panoptic:
        outputs = detections + [model.get_layer(name='semantic')]
    else:
        outputs = detections

    # construct the model
    return Model(inputs=model.inputs, outputs=outputs, name=name)


def RetinaNet(backbone,
              num_classes,
              input_shape,
              norm_method='whole_image',
              location=False,
              use_imagenet=False,
              pooling=None,
              required_channels=3,
              **kwargs):
    """Constructs a retinanet model using a backbone from keras-applications.

    Args:
        backbone: string, name of backbone to use.
        num_classes: Number of classes to classify.
        input_shape: The shape of the input data.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        required_channels: integer, the required number of channels of the
            backbone.  3 is the default for all current backbones.

    Returns:
        RetinaNet model with a backbone.
    """
    inputs = Input(shape=input_shape)
    if location:
        location = Location2D(in_shape=input_shape)(inputs)
        inputs = Concatenate(axis=channel_axis)([inputs, location])

    # force the channel size for backbone input to be `required_channels`
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

    backbone_dict = get_backbone(backbone, fixed_inputs, use_imagenet=use_imagenet, **model_kwargs)

    # create the full model
    return retinanet(
        inputs=inputs,
        num_classes=num_classes,
        backbone_dict=backbone_dict,
        name='{}_retinanet'.format(backbone),
        **kwargs)
