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
"""Tests for ScaleDetectionModel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from deepcell.applications.scale_detection import ScaleDetectionModel
from deepcell.applications import ScaleDetection


class TestScaleDetectionModel(tf.test.TestCase):

    def test_scale_detection_model(self):

        valid_backbones = ['featurenet']
        input_shape = (216, 216, 1)  # channels will be set to 3

        batch_shape = tuple([8] + list(input_shape))

        X = np.random.random(batch_shape)

        for backbone in valid_backbones:
            with self.cached_session():
                inputs = tf.keras.layers.Input(shape=input_shape)
                model = ScaleDetectionModel(
                    inputs=inputs,
                    backbone=backbone)

                y = model.predict(X)

                assert len(y.shape) == 2
                assert y.shape[0] == X.shape[0]

            with self.cached_session():
                model = ScaleDetectionModel(
                    input_shape=input_shape,
                    backbone=backbone)

                y = model.predict(X)

                assert len(y.shape) == 2
                assert y.shape[0] == X.shape[0]


class TestScaleDetection(tf.test.TestCase):

    def test_scale_detection_app(self):
        with self.cached_session():
            model = ScaleDetectionModel(input_shape=(128, 128, 1))
            app = ScaleDetection(model)

            # test output shape
            shape = app.model.output_shape
            self.assertEqual(len(shape), 2)
            self.assertEqual(shape[-1], 1)

            # test predict with default
            x = np.random.rand(1, 500, 500, 1)
            y = app.predict(x)
            self.assertIsInstance(y, np.float32)
