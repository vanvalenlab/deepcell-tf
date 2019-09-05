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
"""Generate cytoplasm segmentations from a phase image"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras

try:
    from tensorflow.python.keras.utils.data_utils import get_file
except ImportError:  # tf v1.9 moves conv_utils from _impl to keras.utils
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file

from deepcell.utils.retinanet_anchor_utils import generate_anchor_params
from deepcell import model_zoo


WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/resnet50_retinanet_20190813_all_phase_512.h5')


def PhaseSegmentationModel(input_shape=(None, None, 1),
                           backbone='resnet50',
                           use_pretrained_weights=True):
    """Initialize a model for cytoplasmic segmentation based on phase data.
    """

    backbone_levels = ['C1', 'C2', 'C3', 'C4', 'C5']
    pyramid_levels = ['P3', 'P4', 'P5', 'P6']
    anchor_size_dicts = {'P3': 32, 'P4': 64, 'P5': 128, 'P6': 256}

    anchor_params = generate_anchor_params(pyramid_levels, anchor_size_dicts)

    model = model_zoo.RetinaMask(
        backbone=backbone,
        use_imagenet=False,
        panoptic=False,
        num_semantic_heads=2,
        num_semantic_classes=[4, 4],
        input_shape=input_shape,
        num_classes=1,
        backbone_levels=backbone_levels,
        pyramid_levels=pyramid_levels,
        anchor_params=anchor_params,
        norm_method='whole_image')

    if use_pretrained_weights:
        if backbone == 'resnet50':
            weights_path = get_file(
                'resnet50_retinanet_20190813_all_phase_512.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='ce31d084fadf7b907a25ab1fcf25529a')

            model.load_weights(weights_path)
        else:
            raise ValueError('Backbone %s does not have a weights file.' %
                             backbone)

    return model
