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
"""A model that can detect whether 2 cells are same, different, or related."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.utils.data_utils import get_file

from deepcell import model_zoo


WEIGHTS_PATH = ('https://deepcell-data.s3-us-west-1.amazonaws.com/'
                'model-weights/tracking_model_benchmarking_757_step5_20'
                'epoch_80split_9tl.h5')


def CellTrackingModel(input_shape=(32, 32, 1),
                      neighborhood_scale_size=30,
                      use_pretrained_weights=True):
    """Creates an instance of a siamese_model used for cell tracking.

    Detects whether to input cells are the same cell, different cells, or
    daughter cells. This can be used along with a cost matrix to track full
    cell lineages across many frames.

    Args:
        input_shape (tuple): a 3-length tuple of the input data shape.
        neighborhood_scale_size (int): size of resized neighborhood images
        use_pretrained_weights (bool): whether to load pre-trained weights.
    """
    features = {'appearance', 'distance', 'neighborhood', 'regionprop'}

    model = model_zoo.siamese_model(
        input_shape=input_shape,
        reg=1e-5,
        init='he_normal',
        neighborhood_scale_size=neighborhood_scale_size,
        features=features)

    if use_pretrained_weights:
        weights_path = get_file(
            'CellTrackingModel.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            md5_hash='3349b363fdad0266a1845ba785e057a6')

        model.load_weights(weights_path)

    return model
