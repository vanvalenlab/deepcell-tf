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
import deepcell_toolbox


class SegmentationApplication(object):
    def __init__(self,
                 model,
                 model_image_shape=(128, 128, 1),
                 dataset_metadata=None,
                 model_metadata=None,
                 model_mpp=0.65,
                 preprocessing_fn=None,
                 postprocessing_fn=None):

        self.model = model

        self.model_image_shape = model_image_shape
        self.model_mpp = model_mpp
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn

    def predict(self, image,
                batch_size=4,
                image_mpp=None,
                preprocess_kwargs={},
                postprocess_kwargs={}):

        # Resize image if necessary

        # Preprocess image
        image = self.preprocessing_fn(image)

        # Tile images
        tiles, tiles_info = deepcell_toolbox.tile_image(image)

        # Run images through model
        output_tiles = self.model.predict(tiles, batch_size=batch_size)

        # Untile images
        output_images = [deepcell_toolbox.untile_image(o, tiles_info) for o in output_tiles]

        # Postprocess predictions to create label image
        label_image = self.postprocessing_fn(output_images, **postprocess_kwargs)

        # Resize label_image back to original resolution if necessary

        return image, tiles, label_image, output_tiles, output_images


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
