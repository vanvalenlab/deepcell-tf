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
"""Cytoplasmic segmentation application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras.utils.data_utils import get_file

from deepcell_toolbox.deep_watershed import deep_watershed

from deepcell.utils.retinanet_anchor_utils import generate_anchor_params
from deepcell import model_zoo
from deepcell.applications import SegmentationApplication
from deepcell.model_zoo import PanopticNet


WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/general_cyto_9c7b79e6238d72c14ea8f87023ac3af9.h5')


class CytoplasmSegmentationApplication(SegmentationApplication):

    def __init__(self,
                 use_pretrained_weights=True,
                 model_image_shape=(128, 128, 1)):

        self.model = PanopticNet('resnet50',
                                 input_shape=model_image_shape,
                                 norm_method='whole_image',
                                 num_semantic_heads=3,
                                 num_semantic_classes=[1, 1, 2],
                                 location=True,
                                 include_top=True)

        if use_pretrained_weights:
            weights_path = get_file(
                os.path.basename(WEIGHTS_PATH),
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='4e9136df5071930a66365b2229fc358b'
            )

            self.model.load_weights(weights_path)
        else:
            weights_path = None

        dataset_metadata = {
            'name': 'general_cyto',
            'other': 'Pooled phase and fluorescent cytoplasm data - computationally curated'
        }

        model_metadata = {
            'batch_size': 2,
            'lr': 1e-5,
            'lr_decay': 0.95,
            'training_seed': 0,
            'n_epochs': 8,
            'training_steps_per_epoch': 7899 // 2,
            'validation_steps_per_epoch': 1973 // 2
        }

        super(CytoplasmSegmentationApplication, self).__init__(self.model,
                                                               model_image_shape=model_image_shape,
                                                               model_mpp=0.65,
                                                               preprocessing_fn=None,
                                                               postprocessing_fn=deep_watershed,
                                                               dataset_metadata=dataset_metadata,
                                                               model_metadata=model_metadata)
