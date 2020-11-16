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
"""Detect the scale of input data for rescaling for other models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from deepcell.applications import Application
from deepcell.layers import ImageNormalization2D
from deepcell.layers import TensorProduct
from deepcell.utils.backbone_utils import get_backbone


MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
              'saved-models/ScaleDetection-1.tar.gz')


def ScaleDetectionModel(input_shape=(None, None, 1),
                        inputs=None,
                        backbone='mobilenetv2'):
    """Create a ``ScaleDetectionModel`` for detecting scales of input data.

    This enables data to be scaled appropriately for other segmentation models
    which may not be resolution tolerant.

    Based on a standard backbone with an intiial ``ImageNormalization2D`` and
    final ``AveragePooling2D`` and ``TensorProduct`` layers.

    Args:
        input_shape (tuple): a 3-length tuple of the input data shape.
        inputs (tensorflow.keras.Layer): Optional input layer of the model.
            If not provided, creates a ``Layer`` based on ``input_shape``.
        backbone (str): name of the backbone to use for the model.
    """
    required_channels = 3  # required for most backbones

    if inputs is None:
        inputs = tf.keras.layers.Input(shape=input_shape)

    if tf.keras.backend.image_data_format() == 'channels_first':
        channel_axis = 0
        raise ValueError('`channels_first` is not supported, '
                         'please use `channels_last`.')
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

    x = tf.keras.layers.AveragePooling2D(4)(backbone_model.outputs[0])
    x = TensorProduct(256, activation='relu')(x)
    x = TensorProduct(1)(x)
    outputs = tf.keras.layers.Flatten()(x)

    model = tf.keras.Model(inputs=backbone_model.inputs, outputs=outputs)

    return model


class ScaleDetection(Application):
    """Loads a :mod:`~ScaleDetectionModel` for detecting relative scales
    of images.

    Args:
        model (tf.keras.Model): The model to load. If ``None``,
            a pre-trained model will be downloaded.
    """

    #: Metadata for the dataset used to train the model
    dataset_metadata = {
        'name': 'general_nuclear_and_cyto_large',
        'other': 'Collection of all available nuclear and cytplasm stains.'
    }

    #: Metadata for the model and training process
    model_metadata = {
        'batch_size': 64,
        'lr': 1e-6,
        'lr_decay': 0.9,
        'training_seed': 0,
        'n_epochs': 200,
        'training_steps_per_epoch': 400,
        'validation_steps_per_epoch': 100
    }

    def __init__(self, model=None):

        if model is None:
            archive_path = tf.keras.utils.get_file(
                'ScaleDetection.tgz', MODEL_PATH,
                file_hash='31f27bd8cc78ee87100041fb292695ff',
                extract=True, cache_subdir='models'
            )
            model_path = os.path.splitext(archive_path)[0]
            model = tf.keras.models.load_model(model_path)

        super(ScaleDetection, self).__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=0.65,
            preprocessing_fn=None,
            postprocessing_fn=None,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)
