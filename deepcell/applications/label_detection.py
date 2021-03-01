# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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

import os

import numpy as np
import tensorflow as tf

from deepcell.applications import Application
from deepcell.layers import ImageNormalization2D
from deepcell.layers import TensorProduct
from deepcell.utils.backbone_utils import get_backbone


MODEL_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
              'saved-models/LabelDetection-1.tar.gz')


def LabelDetectionModel(input_shape=(None, None, 1),
                        inputs=None,
                        backbone='mobilenetv2',
                        num_classes=3):
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
        num_classes (int): The number of labels to detect.
    """
    required_channels = 3  # required for most backbones

    if inputs is None:
        inputs = tf.keras.layers.Input(shape=input_shape)

    if tf.keras.backend.image_data_format() == 'channels_first':
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

    x = tf.keras.layers.AveragePooling2D(4)(backbone_model.outputs[0])
    x = tf.keras.layers.Flatten()(x)
    x = TensorProduct(256)(x)
    x = TensorProduct(num_classes)(x)
    outputs = tf.keras.layers.Softmax(dtype=tf.keras.backend.floatx())(x)

    model = tf.keras.Model(inputs=backbone_model.inputs, outputs=outputs)

    return model


class LabelDetection(Application):
    """Loads a :mod:`~LabelDetectionModel` model for detecting between
    nuclear and cytoplasm.

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
        'lr': 1e-3,
        'lr_decay': 0.9,
        'training_seed': 0,
        'n_epochs': 25,
        'training_steps_per_epoch': 400,
        'validation_steps_per_epoch': 100
    }

    def __init__(self, model=None):

        if model is None:
            archive_path = tf.keras.utils.get_file(
                'LabelDetection.tgz', MODEL_PATH,
                file_hash='eadd047d599e58de91ff4ab1d735f4f0',
                extract=True, cache_subdir='models'
            )
            model_path = os.path.splitext(archive_path)[0]
            model = tf.keras.models.load_model(model_path)

        super(LabelDetection, self).__init__(
            model,
            model_image_shape=model.input_shape[1:],
            model_mpp=0.65,
            preprocessing_fn=None,
            postprocessing_fn=None,
            dataset_metadata=self.dataset_metadata,
            model_metadata=self.model_metadata)

    def predict(self,
                image,
                batch_size=4,
                image_mpp=None):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions.

        Input images are required to have 4 dimensions
        ``[batch, x, y, channel]``.
        Additional empty dimensions can be added using ``np.expand_dims``.

        Args:
            image (numpy.array): Input image with shape
                ``[batch, x, y, channel]``.
            batch_size (int): Number of images to predict on per batch.
            image_mpp (float): Microns per pixel for ``image``.

        Raises:
            ValueError: Input data must match required rank of the application,
                calculated as one dimension more (batch dimension) than expected
                by the model.

            ValueError: Input data must match required number of channels.

        Returns:
            numpy.array: Labeled image
            numpy.array: Model output
        """

        # Check input size of image
        if len(image.shape) != self.required_rank:
            raise ValueError('Input data must have {} dimensions. '
                             'Input data only has {} dimensions'.format(
                                 self.required_rank, len(image.shape)))

        if image.shape[-1] != self.required_channels:
            raise ValueError('Input data must have {} channels. '
                             'Input data only has {} channels'.format(
                                 self.required_channels, image.shape[-1]))

        # Resize image, returns unmodified if appropriate
        resized_image = self._resize_input(image, image_mpp)

        # Tile images, raises error if the image is not 4d
        tiles, _ = self._tile_input(resized_image)

        # Run images through model
        labels = self.model.predict(tiles, batch_size=batch_size)

        labels = np.array(labels)
        vote = labels.sum(axis=0)
        maj = vote.max()

        detected = np.where(vote == maj)[-1][0]
        return detected
