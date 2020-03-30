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
"""Deepcell Applications - Pre-trained models for specific functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

from tensorflow.python.keras import backend as K
from deepcell_toolbox.utils import resize, tile_image, untile_image


class SegmentationApplication(object):
    """Application object that takes a model with weights and manages predictions
    """

    def __init__(self,
                 model,
                 model_image_shape=(128, 128, 1),
                 dataset_metadata=None,
                 model_metadata=None,
                 model_mpp=0.65,
                 preprocessing_fn=None,
                 postprocessing_fn=None):
        """Initializes model for application object

        Args:
            model (tf.model): Tensorflow model with weights loaded
            model_image_shape (tuple, optional): Shape of input expected by model.
                Defaults to (128, 128, 1).
            dataset_metadata (optional): Any input, e.g. str or dict. Defaults to None.
            model_metadata (optional): Any input, e.g. str or dict. Defaults to None.
            model_mpp (float, optional): Microns per pixel resolution of training data.
                Defaults to 0.65.
            preprocessing_fn (function, optional): Preprocessing function to apply to data
                prior to prediction. Defaults to None.
            postprocessing_fn (function, optional): Postprocessing function to apply
                to data after prediction. Defaults to None.
        """

        self.model = model

        self.model_image_shape = model_image_shape
        self.model_mpp = model_mpp
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn

    def predict(self, image,
                batch_size=4,
                image_mpp=None,
                preprocess_kwargs={},
                postprocess_kwargs={},
                debug=False):
        """Generates a labeled image of the input running prediction with
        appropriate pre and post processing functions

        Args:
            image (np.array): Input image.
            batch_size (int, optional): Number of images to predict on per batch. Defaults to 4.
            image_mpp (float, optional): Microns per pixel for the input image. Defaults to None.
            preprocess_kwargs (dict, optional): Kwargs to pass to preprocessing function.
                Defaults to {}.
            postprocess_kwargs (dict, optional): Kwargs to pass to postprocessing function.
                Defaults to {}.
            debug (bool, optional): If True, returns intermediate outputs of data processing.
                Defaults to False.

        Returns:
            np.array: Labeled image, if debug is False.
                If debug is True, returns (image, tiles, output_tiles, output_images, label_image)
        """

        # Check input size of image

        # Resize image if necessary
        if image_mpp != self.model_mpp:
            original_shape = image.shape
            new_shape = np.round(image.shape / (image_mpp / self.model_mpp), decimals=0)
            image = resize(image, new_shape, data_format='channels_last')
        else:
            original_shape = None

        # Preprocess image
        if self.preprocessing_fn is not None:
            image = self.preprocessing_fn(image, **preprocess_kwargs)

        # Tile images, needs 4d
        tiles, tiles_info = tile_image(image)

        # Run images through model
        output_tiles = self.model.predict(tiles, batch_size=batch_size)

        # Untile images
        output_images = [untile_image(o, tiles_info) for o in output_tiles]

        # Postprocess predictions to create label image
        if self.postprocessing_fn is not None:
            label_image = self.postprocessing_fn(output_images, **postprocess_kwargs)
        else:
            label_image = output_images

        # Resize label_image back to original resolution if necessary
        if original_shape is not None:
            label_image = resize(label_image, original_shape, data_format='channels_last')

        if debug:
            return image, tiles, output_tiles, output_images, label_image
        else:
            return label_image


from deepcell.applications.cell_tracking import CellTrackingModel
from deepcell.applications.nuclear_segmentation import NuclearSegmentationModel
from deepcell.applications.label_detection import LabelDetectionModel
from deepcell.applications.scale_detection import ScaleDetectionModel
from deepcell.applications.phase_segmentation import PhaseSegmentationModel
from deepcell.applications.fluorescent_cytoplasm_segmentation import \
    FluorCytoplasmSegmentationModel

del absolute_import
del division
del print_function
