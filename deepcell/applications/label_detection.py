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
"""Classify the type of an input image to send the data to the correct model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.utils import get_file

from deepcell.layers import ImageNormalization2D, TensorProduct
from deepcell.utils.backbone_utils import get_backbone


MOBILENETV2_WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                            'model-weights/LabelDetectionModel_mobilenetv2.h5')


def LabelDetectionModel(input_shape=(None, None, 1),
                        inputs=None,
                        backbone='mobilenetv2',
                        use_pretrained_weights=True):
    """Classify a microscopy image as Nuclear, Cytoplasm, or Phase.

    This can be helpful in determining the type of data (nuclear, cytoplasm,
    etc.) so that this data can be forwared to the correct segmenation model.

    Based on a standard backbone with an intiial ``ImageNormalization2D`` and
    final ``AveragePooling2D``, ``TensorProduct``, and ``Softmax`` layers.

    Args:
        input_shape (tuple): a 3-length tuple of the input data shape.
        inputs (tensorflow.keras.Layer): Optional input layer of the model.
            If not provided, creates a ``Layer`` based on ``input_shape``.
        backbone (str): name of the backbone to use for the model.
        use_pretrained_weights (bool): whether to load pre-trained weights.
            Only supports the ``MobileNetV2`` backbone.
    """
    required_channels = 3  # required for most backbones

    if inputs is None:
        inputs = keras.layers.Input(shape=input_shape)

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 0
    else:
        channel_axis = -1

    norm = ImageNormalization2D(norm_method='whole_image')(inputs)
    fixed_inputs = TensorProduct(required_channels)(norm)

    # force the input shape
    fixed_input_shape = list(input_shape)
    fixed_input_shape[channel_axis] = required_channels
    fixed_input_shape = tuple(fixed_input_shape)

    backbone_model = get_backbone(
        backbone,
        fixed_inputs,
        use_imagenet=False,
        return_dict=False,
        include_top=False,
        weights=None,
        input_shape=fixed_input_shape,
        pooling=None)

    x = keras.layers.AveragePooling2D(4)(backbone_model.outputs[0])
    x = TensorProduct(256)(x)
    x = TensorProduct(3)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Activation('softmax')(x)

    model = keras.Model(inputs=backbone_model.inputs, outputs=outputs)

    if use_pretrained_weights:
        local_name = 'LabelDetectionModel_{}.h5'.format(backbone)
        if backbone.lower() in {'mobilenetv2' or 'mobilenet_v2'}:
            weights_path = get_file(
                local_name,
                MOBILENETV2_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='14d4b2f7c77d334c958d2dde79972e6e')
        else:
            raise ValueError('Backbone %s does not have a weights file.' %
                             backbone)

        model.load_weights(weights_path)

    return model
