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
"""Tests for SegmentationApplication"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras.layers import Input
from tensorflow.python.platform import test

from deepcell.applications import SegmentationApplication


class DummyModel():

    def __init__(self, n_out=1):

        self.n_out = n_out

    def predict(self, x, batch_size=4):

        y = np.random.rand(*x.shape)

        return [y]*self.n_out


class TestSegmentationApplication(test.TestCase):

    def test_predict_resize(self):

        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1)}
        app = SegmentationApplication(model, **kwargs)

        x = np.random.rand(1, 500, 500, 1)

        # image_mpp = None
        y = app.predict(x, image_mpp=None)
        self.assertEqual(x.shape, y.shape)

        # image_mpp = model_mpp
        y = app.predict(x, image_mpp=kwargs['model_mpp'])
        self.assertEqual(x.shape, y.shape)

        # image_mpp > model_mpp
        y = app.predict(x, image_mpp=2.1*kwargs['model_mpp'])
        self.assertEqual(x.shape, y.shape)

        # image_mpp < model_mpp
        y = app.predict(x, image_mpp=0.7*kwargs['model_mpp'])
        self.assertEqual(x.shape, y.shape)

    def test_predict_tiling(self):

        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1)}
        app = SegmentationApplication(model, **kwargs)

        # No tiling
        x = np.random.rand(1, 128, 128, 1)
        y = app.predict(x)
        self.assertEqual(x.shape, y.shape)

        # Tiling square
        x = np.random.rand(1, 400, 400, 1)
        y = app.predict(x)
        self.assertEqual(x.shape, y.shape)

        # Tiling rectangle
        x = np.random.rand(1, 300, 500, 1)
        y = app.predict(x)
        self.assertEqual(x.shape, y.shape)

    def test_predict_input_size(self):

        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1)}
        app = SegmentationApplication(model, **kwargs)

        # Raise valueerror for dim != 4
        with self.assertRaises(ValueError):
            y = app.predict(np.random.rand(128, 128, 1))

    def test_predict_preprocess(self):

        def _preprocess(x):
            y = np.ones(x.shape)
            return y

        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1),
                  'preprocessing_fn': _preprocess}
        app = SegmentationApplication(model, **kwargs)

        x = np.random.rand(1, 300, 300, 1)
        image, tiles, output_tiles, output_images, label_image = app.predict(x, debug=True)
        self.assertEqual(1, len(np.unique(image)))

        # Test bad preprocess input
        kwargs = {'preprocessing_fn': 'x'}
        with self.assertRaises(ValueError):
            app = SegmentationApplication(model, **kwargs)

    def test_predict_postprocess(self):

        def _postprocess(Lx):
            y = np.ones(Lx[0].shape)
            return y

        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1),
                  'postprocessing_fn': _postprocess}
        app = SegmentationApplication(model, **kwargs)

        x = np.random.rand(1, 300, 300, 1)
        y = app.predict(x)
        self.assertEqual(1, len(np.unique(y)))

        # Test bad postprocess input
        kwargs = {'postprocessing_fn': 'x'}
        with self.assertRaises(ValueError):
            app = SegmentationApplication(model, **kwargs)
