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
"""Tests for custom loss functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow import keras
from tensorflow.python.platform import test

from deepcell import losses


ALL_LOSSES = [
    losses.categorical_crossentropy,
    losses.weighted_categorical_crossentropy,
    losses.sample_categorical_crossentropy,
    losses.weighted_focal_loss,
    losses.smooth_l1,
    losses.focal,
    # losses.dice_loss,
    # losses.discriminative_instance_loss
]


class KerasLossesTest(test.TestCase):

    def test_objective_shapes_3d(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.random((5, 6, 7)))
            y_b = keras.backend.variable(np.random.random((5, 6, 7)))
            for obj in ALL_LOSSES:
                objective_output = obj(y_a, y_b)
                self.assertListEqual(objective_output.shape.as_list(), [5, 6])

    def test_objective_shapes_2d(self):
        with self.cached_session():
            y_a = keras.backend.variable(np.random.random((6, 7)))
            y_b = keras.backend.variable(np.random.random((6, 7)))
            for obj in ALL_LOSSES:
                objective_output = obj(y_a, y_b)
                self.assertListEqual(objective_output.shape.as_list(), [6])


if __name__ == '__main__':
    test.main()
