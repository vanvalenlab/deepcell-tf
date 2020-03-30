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
"""Tests for PhaseSegmentationModel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.utils.data_utils import get_file

from deepcell.utils.retinanet_anchor_utils import generate_anchor_params
from deepcell import model_zoo
from deepcell.datasets import SegmentationApplication
from deepcell.model_zoo import PanopticNet
from deepcell.utils.postprocessing_utils import watershed_postprocessing


# WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
#                 'model-weights/resnet50_panoptic_nuclear_segmentation.h5')

WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/general_nuclear_train_batch_size_82800_resnet50_'
                '8_epochs_c4b2167eb754923856bc84fb29074413.h5')


# def NuclearSegmentationModel(input_shape=(None, None, 1),
#                              backbone='resnet50',
#                              use_pretrained_weights=True):
#     """
#     A RetinaMask model with a ResNet50 backbone
#     for nuclear segmentation trained on DAPI data.

#     Args:
#         input_shape (tuple): a 3-length tuple of the input data shape.
#         backbone (str): name of the backbone to use for the model.
#         use_pretrained_weights (bool): whether to load pre-trained weights.
#             Only supports the ResNet50 backbone.
#     """
#     backbone_levels = ['C1', 'C2', 'C3', 'C4', 'C5']
#     pyramid_levels = ['P2', 'P3', 'P4']
#     anchor_size_dict = {'P2': 8, 'P3': 16, 'P4': 32}

#     # Set up the prediction model
#     anchor_params = generate_anchor_params(pyramid_levels, anchor_size_dict)
#     model = model_zoo.RetinaMask(
#         backbone=backbone,
#         use_imagenet=False,
#         panoptic=True,
#         num_semantic_heads=2,
#         num_semantic_classes=[4, 4],
#         input_shape=input_shape,
#         num_classes=1,
#         backbone_levels=backbone_levels,
#         pyramid_levels=pyramid_levels,
#         anchor_params=anchor_params,
#         norm_method='whole_image')

#     if use_pretrained_weights:
#         if backbone == 'resnet50':
#             # '/data/models/resnet50_panoptic_train-val_DVV_V4.h5'
#             weights_path = get_file(
#                 'resnet50_panoptic_nuclear_segmentation.h5',
#                 WEIGHTS_PATH,
#                 cache_subdir='models',
#                 md5_hash='6e925c49cb05a1e3b0e2210220922445')

#             model.load_weights(weights_path)
#         else:
#             raise ValueError('Backbone %s does not have a weights file.' %
#                              backbone)

#     return model


class NuclearSegmentationApplication(SegmentationApplication):

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
                WEIGHTS_PATH.split('/')[-1],
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='eb29808ef2f662fb3bcda6986e47f91a'
            )

            self.model.load_weights(weights_path)

        super(SegmentationApplication, self).__init__(
            self.model,
            weights_path=weights_path,
            model_image_shape=model_image_shape,
            postprocessing_fn=watershed_postprocessing)
