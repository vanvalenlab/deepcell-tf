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
"""Detect the scale of input data for rescaling for other models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.keras.utils.data_utils import get_file

from deepcell.layers import ImageNormalization2D, TensorProduct
from deepcell.utils.backbone_utils import get_backbone


WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/ScaleDetectionModel_VGG16.h5')


def ScaleDetectionModel(input_shape=(None, None, 1),
                        inputs=None,
                        backbone='VGG16',
                        use_pretrained_weights=True):
    """Create a ScaleDetectionModel for detecting scales of input data.

    This enables data to be scaled appropriately for other segmentation models
    which may not be resolution tolerant.
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
    x = TensorProduct(1)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Activation('relu')(x)

    model = keras.Model(inputs=backbone_model.inputs, outputs=outputs)

    if use_pretrained_weights:
        if backbone.upper() == 'VGG16':
            weights_path = get_file(
                'ScaleDetectionModel_{}.h5'.format(backbone),
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='ab23e35676ffcdf1c72d3804cc65ea1d')
        else:
            raise ValueError('Backbone %s does not have a weights file.' %
                             backbone)

        model.load_weights(weights_path)

    return model
