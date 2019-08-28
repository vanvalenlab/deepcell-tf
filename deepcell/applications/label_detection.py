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

from tensorflow.python import keras

from deepcell.layers import ImageNormalization2D, TensorProduct
from deepcell.utils.backbone_utils import get_backbone


def LabelDetectionModel(input_shape=(None, None, 1),
                        inputs=None,
                        backbone='VGG16',
                        required_channels=3,
                        weights=None,
                        norm_method='whole_image',
                        pooling=None):
    """Classify a microscopy image as Nuclear, Cytoplasm, or Phase.

    This can be helpful in determining the type of data (nuclear, cytoplasm,
    etc.) so that this data can be forwared to the correct segmenation model.
    """
    if inputs is None:
        inputs = keras.layers.Input(shape=input_shape)

    norm = ImageNormalization2D(norm_method=norm_method)(inputs)
    fixed_inputs = TensorProduct(required_channels)(norm)

    # force the input shape
    fixed_input_shape = list(input_shape)
    fixed_input_shape[-1] = required_channels
    fixed_input_shape = tuple(fixed_input_shape)

    backbone = get_backbone(
        backbone,
        fixed_inputs,
        use_imagenet=False,
        return_dict=False,
        include_top=False,
        weights=weights,
        input_shape=fixed_input_shape,
        pooling=pooling)

    x = keras.layers.AveragePooling2D(4)(backbone.outputs[0])
    x = TensorProduct(256)(x)
    x = TensorProduct(3)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Activation('softmax')(x)

    return keras.Model(inputs=backbone.inputs, outputs=outputs)
