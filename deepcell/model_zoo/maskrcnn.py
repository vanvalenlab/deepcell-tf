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

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Concatenate
from tensorflow.python.keras.layers import TimeDistributed, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import normal

from deepcell.layers import Cast, Shape
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
        mask_dtype (str): Dtype to use for mask tensors.
        retinanet_dtype (str): Dtype retinanet models expect.
        name (str): The name of the submodel.

    Returns:
        tensorflow.keras.Model: a Model that predicts classes for
            each anchor.
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': normal(mean=0.0, stddev=0.01, seed=None),
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

def default_final_detection_model(pyramid_feature_size=256,
                                  final_detection_feature_size=256,
                                  roi_size=(14,14),
                                  name='final_detection_submodel'):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros',
        'activation': 'relu'
    }
    
    inputs = Input(shape=(None, roi_size[0], roi_size[1], pyramid_feature_size))
    outputs = inputs
    
    for i in range(2):
        outputs = TimeDistributed(Conv2D(
            filters=final_detection_feature_size,
            **options
        ), name='final_detection_submodel_conv1_block{}'.format(i))(outputs)
        outputs = TimeDistributed(Conv2D(
            filters=final_detection_feature_size,
            **options
        ), name='final_detection_submodel_conv2_block{}'.format(i))(outputs)
        outputs = TimeDistributed(MaxPool2D(), 
            name='final_detection_submodel_pool1_block{}'.format(i))(outputs)
        
    outputs = TimeDistributed(Conv2D(filters=final_detection_feature_size, 
                                     kernel_size=3, 
                                     padding='valid',
                                     kernel_initializer=normal(mean=0.0, stddev=0.01, seed=None),
                                     bias_initializer='zeros',
                                     activation='relu'))(outputs)
    
    outputs = TimeDistributed(Conv2D(filters=1, 
                                     kernel_size=1, 
                                     activation='sigmoid'))(outputs)
    
    outputs = Lambda(lambda x: tf.squeeze(x, axis=[2, 3]))(outputs)
    
    return Model(inputs=inputs, outputs=outputs, name=name)
        


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
        ('masks', default_mask_model(num_classes,
                                     roi_size=roi_size,
                                     mask_size=mask_size,
                                     mask_dtype=mask_dtype,
                                     retinanet_dtype=retinanet_dtype)),
    ]


def retinanet_mask(inputs,
                   backbone_dict,
                   num_classes,
                   backbone_levels=['C3', 'C4', 'C5'],
                   pyramid_levels=['P3', 'P4', 'P5', 'P6', 'P7'],
                   retinanet_model=None,
                   anchor_params=None,
                   nms=True,
                   panoptic=False,
                   class_specific_filter=True,
                   crop_size=(14, 14),
                   mask_size=(28, 28),
                   name='retinanet-mask',
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

    if retinanet_model is None:
        retinanet_model = retinanet(
            inputs=image,
            backbone_dict=backbone_dict,
            num_classes=num_classes,
            backbone_levels=backbone_levels,
            pyramid_levels=pyramid_levels,
            panoptic=panoptic,
            num_anchors=anchor_params.num_anchors(),
            **kwargs
        )

    # parse outputs
    regression = retinanet_model.outputs[0]
    classification = retinanet_model.outputs[1]

    if panoptic:
        # Determine the number of semantic heads
        n_semantic_heads = len([1 for layer in retinanet_model.layers if 'semantic' in layer.name])

        # The  panoptic output should not be sent to filter detections
        other = retinanet_model.outputs[2:-n_semantic_heads]
        semantic = retinanet_model.outputs[-n_semantic_heads:]
    else:
        other = retinanet_model.outputs[2:]

    features = [retinanet_model.get_layer(name).output
                for name in pyramid_levels]

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

    # get the region of interest features
    roi_input = [image_shape, boxes, classification] + features
    rois = RoiAlign(crop_size=crop_size)(roi_input)

    # execute maskrcnn submodels
    maskrcnn_outputs = [submodel(rois) for _, submodel in roi_submodels]

    # concatenate boxes for loss computation
    trainable_outputs = [ConcatenateBoxes(name=name)([boxes, output])
                         for (name, _), output in zip(
                             roi_submodels, maskrcnn_outputs)]

    # reconstruct the new output
    outputs = [regression, classification] + other + trainable_outputs + \
        detections + maskrcnn_outputs

    if panoptic:
        outputs += list(semantic)

    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.backbone_levels = backbone_levels
    model.pyramid_levels = pyramid_levels

    return model


def RetinaMask(backbone,
               num_classes,
               input_shape,
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
    inputs = Input(shape=input_shape)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if location:
        location = Location2D(in_shape=input_shape)(inputs)
        concat = Concatenate(axis=channel_axis)([inputs, location])
    else:
        concat = inputs

    # force the channel size for backbone input to be `required_channels`
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

    backbone_dict = get_backbone(backbone, fixed_inputs, use_imagenet=use_imagenet, **model_kwargs)

    # create the full model
    return retinanet_mask(
        inputs=inputs,
        num_classes=num_classes,
        backbone_dict=backbone_dict,
        crop_size=crop_size,
        backbone_levels=backbone_levels,
        pyramid_levels=pyramid_levels,
        name='{}_retinanet_mask'.format(backbone),
        mask_dtype=mask_dtype,
        **kwargs)
