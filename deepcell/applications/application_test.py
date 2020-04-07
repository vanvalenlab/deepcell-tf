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
"""Tests for Application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras.layers import Input
from tensorflow.python.platform import test

from deepcell.applications import Application


class DummyModel():

    def __init__(self, n_out=1):

        self.n_out = n_out

    def predict(self, x, batch_size=4):

        y = np.random.rand(*x.shape)

        return [y] * self.n_out


class TestApplication(test.TestCase):

    def test_predict_notimplemented(self):

        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1)}
        app = Application(model, **kwargs)

        x = np.random.rand(1, 500, 500, 1)

        with self.assertRaises(NotImplementedError):
            app.predict(x)

    def test_resize_input(self):

        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1)}
        app = Application(model, **kwargs)

        x = np.random.rand(1, 500, 500, 1)

        # image_mpp = None --> No resize
        y, original_shape = app._resize_input(x, image_mpp=None)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape, original_shape)

        # image_mpp = model_mpp --> No resize
        y, original_shape = app._resize_input(x, image_mpp=kwargs['model_mpp'])
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.shape, original_shape)

        # image_mpp > model_mpp --> resize
        y, original_shape = app._resize_input(x, image_mpp=2.1 * kwargs['model_mpp'])
        self.assertEqual(2.1, np.round(x.shape[1]/y.shape[1], decimals=1))
        self.assertEqual(x.shape, original_shape)

        # image_mpp < model_mpp --> resize
        y, original_shape = app._resize_input(x, image_mpp=0.7 * kwargs['model_mpp'])
        self.assertEqual(0.7, np.round(x.shape[1]/y.shape[1], decimals=1))
        self.assertEqual(x.shape, original_shape)

    def test_preprocess(self):

        def _preprocess(x):
            y = np.ones(x.shape)
            return y

        model = DummyModel()
        x = np.random.rand(1, 30, 30, 1)

        # Test no preprocess input
        app = Application(model)
        y = app._preprocess(x)
        self.assertAllEqual(x, y)

        # Test ones function
        kwargs = {'preprocessing_fn': _preprocess}
        app = Application(model, **kwargs)
        y = app._preprocess(x)
        self.assertAllEqual(np.ones(x.shape), y)

        # Test bad input
        kwargs = {'preprocessing_fn': 'x'}
        with self.assertRaises(ValueError):
            app = Application(model, **kwargs)

    # def test_tile_input(self):

    # def test_postprocess(self):

    # def test_untile_output(self):

    # def test_resize_output(self):
