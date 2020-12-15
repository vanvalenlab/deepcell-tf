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
"""MaskRCNN models adapted from https://github.com/fizyr/keras-maskrcnn"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import TimeDistributed, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

from deepcell.layers import Cast, UpsampleLike
from deepcell.layers import Upsample, RoiAlign, ConcatenateBoxes
from deepcell.layers import ClipBoxes, RegressBoxes, FilterDetections
from deepcell.layers import TensorProduct, ImageNormalization2D, Location2D
from deepcell.model_zoo.retinanet import retinanet, __build_anchors
from deepcell.utils.retinanet_anchor_utils import AnchorParameters
from deepcell.utils.backbone_utils import get_backbone


def default_mask_model(num_classes,
                       pyramid_feature_size=256,
                       mask_feature_size=256,
                       roi_size=(14, 14),
                       mask_size=(28, 28),
                       name='mask_submodel',
                       mask_dtype=K.floatx(),
                       retinanet_dtype=K.floatx()):
    """Creates the default mask submodel.

    Args:
        num_classes (int): Number of classes to predict a score for at each
            feature level.
        pyramid_feature_size (int): The number of filters to expect from the
            feature pyramid levels.
        mask_feature_size (int): The number of filters to expect from the masks.
        roi_size (tuple): The number of filters to use in the Roi Layers.
        mask_size (tuple): The size of the masks.
        mask_dtype (str): ``dtype`` to use for mask tensors.
        retinanet_dtype (str): ``dtype`` retinanet models expect.
        name (str): The name of the submodel.

    Returns:
        tensorflow.keras.Model: a Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'activation': 'relu',
    }

    inputs = Input(shape=(None, roi_size[0], roi_size[1], pyramid_feature_size))
    outputs = inputs

    # casting to the desidered data type, which may be different than
    # the one used for the underlying keras-retinanet model
    if mask_dtype != retinanet_dtype:
        outputs = TimeDistributed(
            Cast(dtype=mask_dtype),
            name='cast_masks')(outputs)

    for i in range(4):
        outputs = TimeDistributed(Conv2D(
            filters=mask_feature_size,
            **options
        ), name='roi_mask_{}'.format(i))(outputs)

    # perform upsampling + conv instead of deconv as in the paper
    # https://distill.pub/2016/deconv-checkerboard/
    outputs = TimeDistributed(
        Upsample(mask_size),
        name='roi_mask_upsample')(outputs)
    outputs = TimeDistributed(Conv2D(
        filters=mask_feature_size,
        **options
    ), name='roi_mask_features')(outputs)

    outputs = TimeDistributed(Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation='sigmoid'
    ), name='roi_mask')(outputs)

    # casting back to the underlying keras-retinanet model data type
    if mask_dtype != retinanet_dtype:
        outputs = TimeDistributed(
            Cast(dtype=retinanet_dtype),
            name='recast_masks')(outputs)

    return Model(inputs=inputs, outputs=outputs, name=name)


def default_roi_submodels(num_classes,
                          roi_size=(14, 14),
                          mask_size=(28, 28),
                          frames_per_batch=1,
                          mask_dtype=K.floatx(),
                          retinanet_dtype=K.floatx()):
    """Create a list of default roi submodels.

    The default submodels contains a single mask model.

    Args:
        num_classes (int): Number of classes to use.
        roi_size (tuple): The number of filters to use in the Roi Layers.
        mask_size (tuple): The size of the masks.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        mask_dtype (str): ``dtype`` to use for mask tensors.
        retinanet_dtype (str): ``dtype`` retinanet models expect.

    Returns:
        list: A list of tuple, where the first element is the name of the
        submodel and the second element is the submodel itself.
    """
    if frames_per_batch > 1:
        return [
            ('masks', TimeDistributed(
                default_mask_model(num_classes,
                                   roi_size=roi_size,
                                   mask_size=mask_size,
                                   mask_dtype=mask_dtype,
                                   retinanet_dtype=retinanet_dtype,
                                   name='mask_submodel_single_frame'),
                name='mask_submodel'))
        ]
    return [
        ('masks', default_mask_model(num_classes,
                                     roi_size=roi_size,
                                     mask_size=mask_size,
                                     mask_dtype=mask_dtype,
                                     retinanet_dtype=retinanet_dtype))
    ]


def retinamask(inputs,
               backbone_dict,
               num_classes,
               frames_per_batch=1,
               backbone_levels=['C3', 'C4', 'C5'],
               pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
               retinanet_model=None,
               anchor_params=None,
               nms=True,
               training=True,
               panoptic=False,
               class_specific_filter=True,
               crop_size=(14, 14),
               mask_size=(28, 28),
               name='retinanet-mask',
               roi_submodels=None,
               max_detections=100,
               score_threshold=0.05,
               nms_threshold=0.5,
               mask_dtype=K.floatx(),
               **kwargs):
    """Construct a masking model by appending layers to compute masks to a
    :mod:`deepcell.model_zoo.retinanet.retinanet` model.

    Args:
        inputs (tensor): List of ``tensorflow.keras.layers.Input``.
            The first input is the image, the second input the blob of masks.
        backbone_dict (dict): A dictionary with the backbone layers.
        num_classes (int): Integer, number of classes to classify.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        backbone_levels (list): The backbone levels to be used
            to create the feature pyramid.
        pyramid_levels (list): The pyramid levels to attach regression and
            classification heads to.
        retinanet_model (tensorflow.keras.Model):
            :mod:`deepcell.model_zoo.retinanet.retinanet` model that
            predicts regression and classification values.
        anchor_params (AnchorParameters): Struct containing anchor parameters.
        nms (bool): Whether to use non-maximum suppression
            for the filtering step.
        training (bool): Whether to use the bounding boxes as the detections,
            during training or to use the
            :mod:`deepcell.layers.filter_detections.FilterDetections`
            during inference.
        panoptic (bool): Flag for adding the semantic head for panoptic
            segmentation tasks.
        class_specific_filter (bool): Use class specific filtering.
        crop_size (tuple): 2-length tuple for the x-y size of the crops.
            Used to create default ``roi_submodels``.
        mask_size (tuple): 2-length tuple for the x-y size of the masks.
            Used to create default ``roi_submodels``.
        name (str): Name of the model.
        roi_submodels (list): Submodels for processing ROIs.
        max_detections (int): The maximum number of detections allowed.
        score_threshold (float): Minimum score for the
            :mod:`deepcell.layers.filter_detections.FilterDetections` layer.
        nms_threshold (float): Minimimum NMS for the
            :mod:`deepcell.layers.filter_detections.FilterDetections` layer.
        mask_dtype (str): ``dtype`` to use for mask tensors.
        kwargs (dict): Additional kwargs to pass to the
            :mod:`deepcell.model_zoo.retinanet.retinanet` model.

    Returns:
        tensorflow.keras.Model: Model with inputs as input and as output
        the output of each submodel for each pyramid level and the
        detections. The order is as defined in submodels.

        .. code-block:: python

            [
                regression, classification, other[0], ...,
                boxes_masks, boxes, scores, labels, masks, other[0], ...
            ]
    """
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if roi_submodels is None:
        retinanet_dtype = K.floatx()
        K.set_floatx(mask_dtype)
        roi_submodels = default_roi_submodels(
            num_classes, crop_size, mask_size,
            frames_per_batch, mask_dtype, retinanet_dtype)
        K.set_floatx(retinanet_dtype)

    image = inputs
    if retinanet_model is None:
        retinanet_model = retinanet(
            inputs=image,
            backbone_dict=backbone_dict,
            num_classes=num_classes,
            backbone_levels=backbone_levels,
            pyramid_levels=pyramid_levels,
            panoptic=panoptic,
            num_anchors=anchor_params.num_anchors(),
            frames_per_batch=frames_per_batch,
            **kwargs
        )

    # parse outputs
    regression = retinanet_model.outputs[0]
    classification = retinanet_model.outputs[1]
    semantic_classes = [1 for layer in retinanet_model.layers
                        if layer.name.startswith('semantic')]

    if panoptic:
        # Determine the number of semantic heads
        n_semantic_heads = len(semantic_classes)

        # The  panoptic output should not be sent to filter detections
        other = retinanet_model.outputs[2:-n_semantic_heads]
        semantic = retinanet_model.outputs[-n_semantic_heads:]
    else:
        other = retinanet_model.outputs[2:]
        semantic = []

    features = [retinanet_model.get_layer(name).output
                for name in pyramid_levels]

    # build boxes
    anchors = __build_anchors(anchor_params, features,
                              frames_per_batch=frames_per_batch)
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([image, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    if training:
        if frames_per_batch == 1:
            boxes = Input(shape=(None, 4), name='boxes_input')
        else:
            boxes = Input(shape=(None, None, 4), name='boxes_input')
        detections = []

    else:
        detections = FilterDetections(
            nms=nms,
            nms_threshold=nms_threshold,
            score_threshold=score_threshold,
            class_specific_filter=class_specific_filter,
            max_detections=max_detections,
            name='filtered_detections'
        )([boxes, classification] + other)

        # split up in known outputs and "other"
        boxes = detections[0]

    fpn = features[0]
    fpn = UpsampleLike()([fpn, image])
    rois = RoiAlign(crop_size=crop_size)([boxes, fpn])

    # execute maskrcnn submodels
    maskrcnn_outputs = [submodel(rois) for _, submodel in roi_submodels]

    # concatenate boxes for loss computation
    trainable_outputs = [ConcatenateBoxes(name=name)([boxes, output])
                         for (name, _), output in zip(
                             roi_submodels, maskrcnn_outputs)]

    # reconstruct the new output
    outputs = [regression, classification] + other + trainable_outputs + \
        detections + maskrcnn_outputs + list(semantic)

    inputs = [image, boxes] if training else image
    model = Model(inputs=inputs, outputs=outputs, name=name)

    model.backbone_levels = backbone_levels
    model.pyramid_levels = pyramid_levels
    return model


def retinamask_bbox(model,
                    nms=True,
                    panoptic=False,
                    num_semantic_heads=1,
                    class_specific_filter=True,
                    name='retinanet-bbox',
                    anchor_params=None,
                    max_detections=300,
                    frames_per_batch=1,
                    crop_size=(14, 14),
                    **kwargs):
    """Construct a RetinaNet model on top of a backbone and adds convenience
    functions to output boxes directly.
    This model uses the minimum retinanet model and appends a few layers
    to compute boxes within the graph. These layers include applying the
    regression values to the anchors and performing NMS.

    Args:
        model (tensorflow.keras.Model): RetinaNet model to append bbox
            layers to. If ``None``, it will create a ``RetinaNet`` model
            using ``kwargs``.
        nms (bool): Whether to use non-maximum suppression
            for the filtering step.
        panoptic (bool): Flag for adding the semantic head for panoptic
            segmentation tasks.
        num_semantic_heads (int): Total number of semantic heads to build.
        class_specific_filter (bool): Whether to use class specific filtering
            or filter for the best scoring class only.
        anchor_params (AnchorParameters): Struct containing anchor parameters.
        max_detections (int): The maximum number of detections allowed.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        crop_size (tuple): 2-length tuple for the x-y size of the crops.
            Used to create default ``roi_submodels``.
        kwargs (dict): Additional kwargs to pass to the
            :mod:`deepcell.model_zoo.retinanet.retinanet` model.

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
    semantic_classes = [1 for layer in model.layers
                        if layer.name.startswith('semantic')]

    # "other" can be any additional output from custom submodels, by default []
    if panoptic:
        # The last output is the panoptic output, which should not be
        # sent to filter detections
        num_semantic_heads = len(semantic_classes)
        other = model.outputs[2:-num_semantic_heads]
        semantic = model.outputs[-num_semantic_heads:]
    else:
        other = model.outputs[2:]
        semantic = []

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        max_detections=max_detections,
        name='filtered_detections'
    )([boxes, classification])

    # apply submodels to detections
    image = model.layers[0].output
    boxes = detections[0]

    fpn = features[0]
    fpn = UpsampleLike()([fpn, image])
    rois = RoiAlign(crop_size=crop_size)([boxes, fpn])

    mask_submodel = model.get_layer('mask_submodel')
    masks = [mask_submodel(rois)]

    # add the semantic head's output if needed
    outputs = detections + list(masks) + list(semantic)

    # construct the model
    new_model = Model(inputs=model.inputs, outputs=outputs, name=name)

    image_input = model.inputs[0]
    shape = (1, 1, 4) if frames_per_batch == 1 else (1, 1, 1, 4)
    temp_boxes = K.zeros(shape, name='temp_boxes')
    new_inputs = [image_input, temp_boxes]

    final_model = new_model(new_inputs)
    return Model(inputs=image_input, outputs=final_model)


def RetinaMask(backbone,
               num_classes,
               input_shape,
               inputs=None,
               backbone_levels=['C3', 'C4', 'C5'],
               pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
               norm_method='whole_image',
               location=False,
               use_imagenet=False,
               crop_size=(14, 14),
               pooling=None,
               mask_dtype=K.floatx(),
               required_channels=3,
               frames_per_batch=1,
               **kwargs):
    """Constructs a mrcnn model using a backbone from ``keras-applications``.

    Args:
        backbone (str): Name of backbone to use.
        num_classes (int): Number of classes to classify.
        input_shape (tuple): The shape of the input data.
        inputs (tensor): Optional input tensor, overrides ``input_shape``.
        backbone_levels (list): The backbone levels to be used.
            to create the feature pyramid.
        pyramid_levels (list): The pyramid levels to attach regression and
            classification heads.
        norm_method (str): Normalization method to use with the
            :mod:`deepcell.layers.normalization.ImageNormalization2D` layer.
        location (bool): Whether to include a
            :mod:`deepcell.layers.location.Location2D` layer.
        use_imagenet (bool): Whether to load imagenet-based
            pretrained weights.
        crop_size (tuple): 2-length tuple for the x-y size of the crops.
            Used to create default ``roi_submodels``.
        pooling (str): Pooling mode for feature extraction
            when ``include_top`` is ``False``.

            - None means that the output of the model will be
              the 4D tensor output of the last convolutional layer.
            - 'avg' means that global average pooling will be applied to
              the output of the last convolutional layer, and thus
              the output of the model will be a 2D tensor.
            - 'max' means that global max pooling will be applied.

        mask_dtype (str): ``dtype`` to use for mask tensors.
        required_channels (int): The required number of channels of the
            backbone.  3 is the default for all current backbones.
        frames_per_batch (int): Size of z axis in generated batches.
            If equal to 1, assumes 2D data.
        kwargs (dict): Other standard inputs for `~retinamask`.

    Returns:
        tensorflow.keras.Model: :mod:`deepcell.model_zoo.retinanet.RetinaNet`
        model with additional mask output.
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
    return retinamask(
        inputs=inputs,
        num_classes=num_classes,
        backbone_dict=backbone_dict,
        crop_size=crop_size,
        backbone_levels=backbone_levels,
        pyramid_levels=pyramid_levels,
        name='{}_retinanetmask'.format(backbone),
        mask_dtype=mask_dtype,
        frames_per_batch=frames_per_batch,
        **kwargs)
