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
"""Tests for NuclearSegmentationApplication"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
import numpy as np

from deepcell.model_zoo import PanopticNet
from deepcell.applications import NuclearSegmentation


class TestNuclearSegmentation(test.TestCase):

    def test_nuclear_app(self):
        with self.cached_session():
            model = PanopticNet(
                'resnet50',
                input_shape=(128, 128, 1),
                norm_method='whole_image',
                num_semantic_heads=2,
                num_semantic_classes=[1, 1],
                location=True,
                include_top=True,
                lite=True,
                use_imagenet=False,
                interpolation='bilinear')
            app = NuclearSegmentation(model)

            # test output shape
            shape = app.model.output_shape
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 2)
            self.assertEqual(len(shape[0]), 4)
            self.assertEqual(len(shape[1]), 4)

            # test predict
            x = np.random.rand(1, 500, 500, 1)
            y = app.predict(x)
            self.assertEqual(x.shape, y.shape)
