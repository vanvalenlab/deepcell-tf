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
"""MaskRCNN models adapted from https://github.com/fizyr/keras-maskrcnn"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Concatenate, Lambda, Dense
from tensorflow.python.keras.layers import Layer, Softmax, TimeDistributed
from tensorflow.python.keras.models import Model

from deepcell.layers import Shape, UpsampleLike, RoiAlign, ConcatenateBoxes, RoiAlign3D
from deepcell.layers import ClipBoxes, RegressBoxes, FilterDetections
from deepcell.layers import TensorProduct, ImageNormalization2D, Location2D
from deepcell.model_zoo.retinamovie import retinamovie
from deepcell.model_zoo.maskrcnn import default_mask_model, default_final_detection_model
from deepcell.model_zoo.retinamovie import __build_anchors
from deepcell.utils.retinanet_anchor_utils import AnchorParameters
from deepcell.utils.backbone_utils import get_backbone


def default_roi_submodels(num_classes,
                          roi_size=(14, 14),
                          mask_size=(28, 28),
                          mask_dtype=K.floatx(),
                          retinanet_dtype=K.floatx()):
    """Create a list of default roi submodels.

    The default submodels contains a single mask model.

    Args:
        num_classes (int): Number of classes to use.
        roi_size (tuple): The number of filters to use in the Roi Layers.
        mask_size (tuple): The size of the masks.
        mask_dtype (str): Dtype to use for mask tensors.
        retinanet_dtype (str): Dtype retinanet models expect.

    Returns:
        list: A list of tuple, where the first element is the name of the
            submodel and the second element is the submodel itself.
    """
    return [
        ('masks', TimeDistributed(
            default_mask_model(num_classes,
                               roi_size=roi_size,
                               mask_size=mask_size,
                               mask_dtype=mask_dtype,
                               retinanet_dtype=retinanet_dtype))),
        ('final_detection', TimeDistributed(
            default_final_detection_model(roi_size=roi_size)))
    ]


def retinamovie_mask(inputs,
                     backbone_dict,
                     num_classes,
                     frames_per_batch=5,
                     backbone_levels=['C3', 'C4', 'C5'],
                     pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                     retinamovie_model=None,
                     anchor_params=None,
                     nms=True,
                     class_specific_filter=True,
                     crop_size=(14, 14),
                     mask_size=(28, 28),
                     name='retinamovie-mask',
                     roi_submodels=None,
                     max_detections=100,
                     mask_dtype=K.floatx(),
                     **kwargs):
    """Construct a RetinaNet mask model on top of a retinanet bbox model.
    Uses the retinanet bbox model and appends layers to compute masks.

    Args:
        inputs (tensor): List of tensorflow.keras.layers.Input.
            The first input is the image, the second input the blob of masks.
        num_classes (int): Integer, number of classes to classify.
        retinanet_model (tensorflow.keras.Model): RetinaNet model that predicts
            regression and classification values.
        anchor_params (AnchorParameters): Struct containing anchor parameters.
        nms (bool): Whether to use NMS.
        class_specific_filter (bool): Use class specific filtering.
        roi_submodels (list): Submodels for processing ROIs.
        name (str): Name of the model.
        mask_dtype (str): Dtype to use for mask tensors.
        kwargs (dict): Additional kwargs to pass to the retinanet bbox model.

    Returns:
        tensorflow.keras.Model: Model with inputs as input and as output
            the output of each submodel for each pyramid level and the
            detections. The order is as defined in submodels.

            ```
            [
                regression, classification, other[0], ...,
                boxes_masks, boxes, scores, labels, masks, other[0], ...
            ]
            ```
    """
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if roi_submodels is None:
        retinanet_dtype = K.floatx()
        K.set_floatx(mask_dtype)
        roi_submodels = default_roi_submodels(
            num_classes, crop_size, mask_size,
            mask_dtype, retinanet_dtype)
        K.set_floatx(retinanet_dtype)

    image = inputs
    image_shape = Shape()(image)

    if retinamovie_model is None:
        retinamovie_model = retinamovie(
            inputs=image,
            frames_per_batch=frames_per_batch,
            backbone_dict=backbone_dict,
            num_classes=num_classes,
            backbone_levels=backbone_levels,
            pyramid_levels=pyramid_levels,
            num_anchors=anchor_params.num_anchors(),
            **kwargs
        )

    # parse outputs
    regression = retinamovie_model.outputs[0]
    classification = retinamovie_model.outputs[1]
    other = retinamovie_model.outputs[2:]

    features = [retinamovie_model.get_layer(l).output for l in retinamovie_model.pyramid_levels]

    # build boxes
    anchors = __build_anchors(anchor_params, features)
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([image, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        max_detections=max_detections,
        name='filtered_detections'
    )([boxes, classification] + other)

    # split up in known outputs and "other"
    boxes = detections[0]
    scores = detections[1]

    fpn = features[0]
    fpn = UpsampleLike()([fpn, image])

    rois = RoiAlign3D()([boxes, fpn])

    # execute maskrcnn submodels
    maskrcnn_outputs = [submodel(rois) for _, submodel in roi_submodels]

    # concatenate boxes for loss computation
    trainable_outputs = [ConcatenateBoxes(name=name)([boxes, output])
                         for (name, _), output in zip(
                             roi_submodels, maskrcnn_outputs)]

    # reconstruct the new output
    outputs = [regression, classification] + other + trainable_outputs + \
        detections + maskrcnn_outputs

    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.backbone_levels = backbone_levels
    model.pyramid_levels = pyramid_levels

    return model


def RetinaMovieMask(backbone,
                    num_classes,
                    input_shape,
                    frames_per_batch=5,
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
                    **kwargs):
    """Constructs a mrcnn model using a backbone from keras-applications.

    Args:
        backbone (str): Name of backbone to use.
        num_classes (int): Number of classes to classify.
        input_shape (tuple): The shape of the input data.
        weights (str): one of None (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
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

    Returns:
        tensorflow.keras.Model: RetinaNet model with a backbone.
    """
    if inputs is None:
        input_shape_with_time = tuple([frames_per_batch] + list(input_shape))
        inputs = Input(shape=input_shape_with_time)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if location:
        location = TimeDistributed(Location2D(in_shape=input_shape))(inputs)
        concat = Concatenate(axis=channel_axis)([inputs, location])
    else:
        concat = inputs

    # force the channel size for backbone input to be `required_channels`
    norm = TimeDistributed(ImageNormalization2D(norm_method=norm_method))(concat)
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

    backbone_dict = get_backbone(backbone,
                                 input_tensor=fixed_inputs,
                                 use_imagenet=use_imagenet,
                                 time_distribute=True,
                                 frames_per_batch=frames_per_batch,
                                 **model_kwargs)

    # create the full model
    return retinamovie_mask(
        inputs=inputs,
        frames_per_batch=frames_per_batch,
        num_classes=num_classes,
        backbone_dict=backbone_dict,
        crop_size=crop_size,
        backbone_levels=backbone_levels,
        pyramid_levels=pyramid_levels,
        name='{}_retinanet_mask'.format(backbone),
        mask_dtype=mask_dtype,
        **kwargs)
