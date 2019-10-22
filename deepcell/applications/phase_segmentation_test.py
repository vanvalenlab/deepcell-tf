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

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import test

from deepcell.applications import PhaseSegmentationModel


class TestPhaseSegmentationModel(test.TestCase):

    def test_phase_segmentation_model(self):

        # TODO: resnet50 is trained but tests fail due to tf version
        valid_backbones = ['featurenet']
        input_shape = (256, 256, 1)  # channels will be set to 3

        batch_shape = tuple([8] + list(input_shape))

        X = np.random.random(batch_shape)

        for backbone in valid_backbones:
            if int(tf.VERSION.split('.')[1]) < 10:
                # retinanet backbones do not work with versions < 1.10.0
                continue

            with self.test_session():
                model = PhaseSegmentationModel(
                    input_shape=input_shape,
                    backbone=backbone,
                    use_pretrained_weights=False
                )

                shape = model.output_shape

                self.assertIsInstance(shape, list)
                self.assertEqual(shape[0][-1], 4)  # bounding boxes
                self.assertEqual(shape[1][-1], 1)  # labels
                self.assertEqual(shape[6][-3:-1], (28, 28))  # maskRCNN output
                self.assertEqual(len(shape), 7)  # maskRCNN output is 7
