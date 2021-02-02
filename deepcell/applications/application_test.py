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
"""Tests for Application"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.keras.layers import Input
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

    def test_resize(self):
        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1)}
        app = Application(model, **kwargs)

        x = np.random.rand(1, 500, 500, 1)

        # image_mpp = None --> No resize
        y = app._resize_input(x, image_mpp=None)
        self.assertEqual(x.shape, y.shape)

        # image_mpp = model_mpp --> No resize
        y = app._resize_input(x, image_mpp=kwargs['model_mpp'])
        self.assertEqual(x.shape, y.shape)

        # image_mpp > model_mpp --> resize
        y = app._resize_input(x, image_mpp=2.1 * kwargs['model_mpp'])
        self.assertEqual(2.1, np.round(y.shape[1] / x.shape[1], decimals=1))

        # image_mpp < model_mpp --> resize
        y = app._resize_input(x, image_mpp=0.7 * kwargs['model_mpp'])
        self.assertEqual(0.7, np.round(y.shape[1] / x.shape[1], decimals=1))

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

    def test_tile_input(self):
        model = DummyModel()
        kwargs = {'model_mpp': 0.65,
                  'model_image_shape': (128, 128, 1)}
        app = Application(model, **kwargs)

        # No tiling
        x = np.random.rand(1, 128, 128, 1)
        y, tile_info = app._tile_input(x)
        self.assertEqual(x.shape, y.shape)
        self.assertIsInstance(tile_info, dict)

        # Tiling square
        x = np.random.rand(1, 400, 400, 1)
        y, tile_info = app._tile_input(x)
        self.assertEqual(kwargs['model_image_shape'][:-1], y.shape[1:-1])
        self.assertIsInstance(tile_info, dict)

        # Tiling rectangle
        x = np.random.rand(1, 300, 500, 1)
        y, tile_info = app._tile_input(x)
        self.assertEqual(kwargs['model_image_shape'][:-1], y.shape[1:-1])
        self.assertIsInstance(tile_info, dict)

        # Smaller than expected
        x = np.random.rand(1, 100, 100, 1)
        y, tile_info = app._tile_input(x)
        self.assertEqual(kwargs['model_image_shape'][:-1], y.shape[1:-1])
        self.assertIsInstance(tile_info, dict)

    def test_postprocess(self):

        def _postprocess(Lx):
            y = np.ones(Lx[0].shape)
            return y

        model = DummyModel()
        x = np.random.rand(1, 30, 30, 1)

        # No input
        app = Application(model)
        y = app._postprocess(x)
        self.assertAllEqual(x, y)

        # Ones
        kwargs = {'postprocessing_fn': _postprocess}
        app = Application(model, **kwargs)
        y = app._postprocess([x])
        self.assertAllEqual(np.ones(x.shape), y)

        # Bad input
        kwargs = {'postprocessing_fn': 'x'}
        with self.assertRaises(ValueError):
            app = Application(model, **kwargs)

    def test_untile_output(self):
        model = DummyModel()
        kwargs = {'model_image_shape': (128, 128, 1)}
        app = Application(model, **kwargs)

        # No tiling
        x = np.random.rand(1, 128, 128, 1)
        tiles, tile_info = app._tile_input(x)
        y = app._untile_output(tiles, tile_info)
        self.assertEqual(x.shape, y.shape)

        # Tiling square
        x = np.random.rand(1, 400, 400, 1)
        tiles, tile_info = app._tile_input(x)
        y = app._untile_output(tiles, tile_info)
        self.assertEqual(x.shape, y.shape)

        # Tiling rectangle
        x = np.random.rand(1, 300, 500, 1)
        tiles, tile_info = app._tile_input(x)
        y = app._untile_output(tiles, tile_info)
        self.assertEqual(x.shape, y.shape)

        # Smaller than expected
        x = np.random.rand(1, 100, 100, 1)
        tiles, tile_info = app._tile_input(x)
        y = app._untile_output(tiles, tile_info)
        self.assertEqual(x.shape, y.shape)

    def test_resize_output(self):
        model = DummyModel()
        kwargs = {'model_image_shape': (128, 128, 1)}
        app = Application(model, **kwargs)

        x = np.random.rand(1, 128, 128, 1)

        # x.shape = original_shape --> no resize
        y = app._resize_output(x, x.shape)
        self.assertEqual(x.shape, y.shape)

        # x.shape != original_shape --> resize
        original_shape = (1, 500, 500, 1)
        y = app._resize_output(x, original_shape)
        self.assertEqual(original_shape, y.shape)

        # test multiple outputs are also resized
        x_list = [x, x]

        # x.shape = original_shape --> no resize
        y = app._resize_output(x_list, x.shape)
        self.assertIsInstance(y, list)
        for y_sub in y:
            self.assertEqual(x.shape, y_sub.shape)

        # x.shape != original_shape --> resize
        original_shape = (1, 500, 500, 1)
        y = app._resize_output(x_list, original_shape)
        self.assertIsInstance(y, list)
        for y_sub in y:
            self.assertEqual(original_shape, y_sub.shape)

    def test_format_model_output(self):
        def _format_model_output(Lx):
            return {'inner-distance': Lx}

        model = DummyModel()
        x = np.random.rand(1, 30, 30, 1)

        # No function
        app = Application(model)
        y = app._format_model_output(x)
        self.assertAllEqual(x, y)

        # single image
        kwargs = {'format_model_output_fn': _format_model_output}
        app = Application(model, **kwargs)
        y = app._format_model_output(x)
        self.assertAllEqual(x, y['inner-distance'])

    def test_run_model(self):
        model = DummyModel()
        app = Application(model)

        x = np.random.rand(1, 128, 128, 1)
        y = app._run_model(x)
        self.assertEqual(x.shape, y[0].shape)

    def test_predict_segmentation(self):
        model = DummyModel()
        app = Application(model)

        x = np.random.rand(1, 128, 128, 1)
        y = app._predict_segmentation(x)
        self.assertEqual(x.shape, y.shape)

        # test with different MPP
        model = DummyModel()
        app = Application(model)

        x = np.random.rand(1, 128, 128, 1)
        y = app._predict_segmentation(x, image_mpp=1.3)
        self.assertEqual(x.shape, y.shape)
