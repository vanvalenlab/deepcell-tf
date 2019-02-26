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

from tensorflow.python.keras import applications
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import TimeDistributed, Conv2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import normal

from deepcell.layers import Cast, Shape
from deepcell.layers import Upsample, RoiAlign, ConcatenateBoxes
from deepcell.layers import ClipBoxes, RegressBoxes, FilterDetections
from deepcell.layers import TensorProduct, ImageNormalization2D
from deepcell.model_zoo.retinanet import retinanet, __build_anchors
from deepcell.utils.retinanet_anchor_utils import AnchorParameters


def default_mask_model(num_classes,
                       pyramid_feature_size=256,
                       mask_feature_size=256,
                       roi_size=(14, 14),
                       mask_size=(28, 28),
                       name='mask_submodel',
                       mask_dtype=K.floatx(),
                       retinanet_dtype=K.floatx()):

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


def default_roi_submodels(num_classes,
                          mask_dtype=K.floatx(),
                          retinanet_dtype=K.floatx()):
    return [
        ('masks', default_mask_model(num_classes, mask_dtype=mask_dtype,
                                     retinanet_dtype=retinanet_dtype)),
    ]


def retinanet_mask(inputs,
                   num_classes,
                   retinanet_model=None,
                   anchor_params=None,
                   nms=True,
                   class_specific_filter=False,
                   name='retinanet-mask',
                   roi_submodels=None,
                   mask_dtype=K.floatx(),
                   **kwargs):
    """Construct a RetinaNet mask model on top of a retinanet bbox model.

    Uses the retinanet bbox model and appends layers to compute masks.

    Args:
        inputs: List of tensorflow.keras.layers.Input.
            The first input is the image, the second input the blob of masks.
        num_classes: Integer, number of classes to classify.
        retinanet_model: deepcell.model_zoo.retinanet.retinanet model,
            returning regression and classification values.
        anchor_params: Struct containing anchor parameters.
        nms: Boolean, whether to use NMS.
        class_specific_filter: Boolean, use class specific filtering.
        roi_submodels: Submodels for processing ROIs.
        mask_dtype: Data type of the masks, can be different from the main one.
        name: Name of the model.
        **kwargs: Additional kwargs to pass to the retinanet bbox model.
    Returns:
        Model with inputs as input and as output the output of each submodel
        for each pyramid level and the detections. The order is as defined in
        submodels.
        ```
        [
            regression, classification, other[0], other[1], ...,
            boxes_masks, boxes, scores, labels, masks, other[0], other[1], ...
        ]
        ```
    """
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if roi_submodels is None:
        retinanet_dtype = K.floatx()
        K.set_floatx(mask_dtype)
        roi_submodels = default_roi_submodels(
            num_classes, mask_dtype, retinanet_dtype)
        K.set_floatx(retinanet_dtype)

    image = inputs
    image_shape = Shape()(image)

    if retinanet_model is None:
        retinanet_model = retinanet(
            inputs=image,
            num_classes=num_classes,
            num_anchors=anchor_params.num_anchors(),
            **kwargs
        )

    # parse outputs
    regression = retinanet_model.outputs[0]
    classification = retinanet_model.outputs[1]
    other = retinanet_model.outputs[2:]
    features = [retinanet_model.get_layer(name).output
                for name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    # build boxes
    anchors = __build_anchors(anchor_params, features)
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([image, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        max_detections=100,
        name='filtered_detections'
    )([boxes, classification] + other)

    # split up in known outputs and "other"
    boxes = detections[0]
    scores = detections[1]

    # get the region of interest features
    rois = RoiAlign()([image_shape, boxes, scores] + features)

    # execute maskrcnn submodels
    maskrcnn_outputs = [submodel(rois) for _, submodel in roi_submodels]

    # concatenate boxes for loss computation
    zipped = zip(roi_submodels, maskrcnn_outputs)
    trainable_outputs = [ConcatenateBoxes(name=name)([boxes, output])
                         for (name, _), output in zipped]

    # reconstruct the new output
    outputs = [regression, classification]
    outputs += other + trainable_outputs + detections + maskrcnn_outputs

    return Model(inputs=inputs, outputs=outputs, name=name)


def MaskRCNN(backbone,
             num_classes,
             input_shape,
             norm_method='whole_image',
             weights=None,
             pooling=None,
             required_channels=3,
             **kwargs):
    """Constructs a mrcnn model using a backbone from keras-applications.

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
    # force the channel size for backbone input to be `required_channels`
    norm = ImageNormalization2D(norm_method=norm_method)(inputs)
    fixed_inputs = TensorProduct(required_channels)(norm)
    model_kwargs = {
        'include_top': False,
        'input_tensor': fixed_inputs,
        'weights': weights,
        'pooling': pooling
    }
    vgg_backbones = {'vgg16', 'vgg19'}
    densenet_backbones = {'densenet121', 'densenet169', 'densenet201'}
    mobilenet_backbones = {'mobilenet', 'mobilenet_v2'}
    resnet_backbones = {'resnet50'}
    nasnet_backbones = {'nasnet_large', 'nasnet_mobile'}

    if backbone in vgg_backbones:
        layer_names = ['block3_pool', 'block4_pool', 'block5_pool']
        if backbone == 'vgg16':
            model = applications.VGG16(**model_kwargs)
        else:
            model = applications.VGG19(**model_kwargs)
        layer_outputs = [model.get_layer(n).output for n in layer_names]

    elif backbone in densenet_backbones:
        if backbone == 'densenet121':
            model = applications.DenseNet121(**model_kwargs)
            blocks = [6, 12, 24, 16]
        elif backbone == 'densenet169':
            model = applications.DenseNet169(**model_kwargs)
            blocks = [6, 12, 32, 32]
        elif backbone == 'densenet201':
            model = applications.DenseNet201(**model_kwargs)
            blocks = [6, 12, 48, 32]
        layer_outputs = []
        for idx, block_num in enumerate(blocks):
            name = 'conv{}_block{}_concat'.format(idx + 2, block_num)
            layer_outputs.append(model.get_layer(name=name).output)
        # create the densenet backbone
        model = Model(inputs=inputs, outputs=layer_outputs[1:], name=model.name)
        layer_outputs = model.outputs

    elif backbone in resnet_backbones:
        model = applications.ResNet50(**model_kwargs)
        layer_names = ['res3d_branch2c', 'res4f_branch2c', 'res5c_branch2c']
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        model = Model(inputs=inputs, outputs=layer_outputs, name=model.name)
        layer_outputs = model.outputs

    elif backbone in mobilenet_backbones:
        alpha = kwargs.get('alpha', 1.0)
        if backbone.endswith('v2'):
            model = applications.MobileNetV2(alpha=alpha, **model_kwargs)
            block_ids = (12, 15, 16)
            layer_names = ['block_%s_depthwise_relu' % i for i in block_ids]
        else:
            model = applications.MobileNet(alpha=alpha, **model_kwargs)
            block_ids = (5, 11, 13)
            layer_names = ['conv_pw_%s_relu' % i for i in block_ids]
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        model = Model(inputs=inputs, outputs=layer_outputs, name=model.name)
        layer_outputs = model.outputs

    elif backbone in nasnet_backbones:
        if backbone.endswith('large'):
            model = applications.NASNetLarge(**model_kwargs)
            block_ids = [5, 12, 18]
        else:
            model = applications.NASNetMobile(**model_kwargs)
            block_ids = [3, 8, 12]
        layer_names = ['normal_conv_1_%s' % i for i in block_ids]
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        model = Model(inputs=inputs, outputs=layer_outputs, name=model.name)
        layer_outputs = model.outputs

    else:
        backbones = list(densenet_backbones + resnet_backbones + vgg_backbones)
        raise ValueError('Invalid value for `backbone`. Must be one of: %s' %
                         ', '.join(backbones))

    # create the full model
    return retinanet_mask(
        inputs=inputs,
        num_classes=num_classes,
        backbone_layers=layer_outputs,
        **kwargs)
