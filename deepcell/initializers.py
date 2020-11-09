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
"""Custom initializers"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Initializer


class PriorProbability(Initializer):
    """Initializer that applies a prior probability to the weights.

    Adapted from https://github.com/fizyr/keras-retinanet.

    Args:
        probability (float): The prior probability to apply to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None, partition_info=None):
        # set bias to -log((1 - p)/p) for foreground
        bias = -K.log((1 - self.probability) / self.probability)
        result = K.get_value(K.ones(shape, dtype=dtype)) * bias
        return result
