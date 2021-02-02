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
"""Tests for MultiplexSegmentationApplication"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
import numpy as np

from deepcell.model_zoo import PanopticNet
from deepcell.applications import MultiplexSegmentation


class TestMultiplexSegmentation(test.TestCase):

    def test_multiplex_app(self):
        with self.cached_session():
            whole_cell_classes = [1, 3]
            nuclear_classes = [1, 3]
            num_semantic_classes = whole_cell_classes + nuclear_classes
            num_semantic_heads = len(num_semantic_classes)

            model = PanopticNet(
                'resnet50',
                input_shape=(256, 256, 2),
                norm_method=None,
                num_semantic_heads=num_semantic_heads,
                num_semantic_classes=num_semantic_classes,
                location=True,
                include_top=True,
                use_imagenet=False)

            app = MultiplexSegmentation(model)

            # test output shape
            shape = app.model.output_shape
            self.assertIsInstance(shape, list)
            self.assertEqual(len(shape), 4)

            # test predict with default
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x)
            self.assertEqual(x.shape[:-1], y.shape[:-1])

            # test predict with nuclear compartment only
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x, compartment='nuclear')
            self.assertEqual(x.shape[:-1], y.shape[:-1])
            self.assertEqual(y.shape[-1], 1)

            # test predict with cell compartment only
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x, compartment='whole-cell')
            self.assertEqual(x.shape[:-1], y.shape[:-1])
            self.assertEqual(y.shape[-1], 1)

            # test predict with both cell and nuclear compartments
            x = np.random.rand(1, 500, 500, 2)
            y = app.predict(x, compartment='both')
            self.assertEqual(x.shape[:-1], y.shape[:-1])
            self.assertEqual(y.shape[-1], 2)
