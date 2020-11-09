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
"""Tests for the retinanet layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.keras.utils import custom_object_scope
from tensorflow.python.platform import test

from deepcell import layers


@keras_parameterized.run_all_keras_modes
class TestAnchors(keras_parameterized.TestCase):

    def test_anchors_2d(self):
        with custom_object_scope({'Anchors': layers.Anchors}):
            testing_utils.layer_test(
                layers.Anchors,
                kwargs={'size': 1, 'stride': 1,
                        'data_format': 'channels_last'},
                input_shape=(3, 5, 6, 4))
            testing_utils.layer_test(
                layers.Anchors,
                kwargs={'size': 1, 'stride': 1,
                        'data_format': 'channels_last'},
                input_shape=(3, None, None, None))
            testing_utils.layer_test(
                layers.Anchors,
                kwargs={'size': 1, 'stride': 1,
                        'data_format': 'channels_first'},
                input_shape=(3, 5, 6, 4))

    def test_simple(self):
        # create simple Anchors layer
        anchors_layer = layers.Anchors(
            size=32,
            stride=8,
            ratios=np.array([1], K.floatx()),
            scales=np.array([1], K.floatx()),
        )

        # create fake features input (only shape is used anyway)
        features = np.zeros((1, 2, 2, 1024), dtype=K.floatx())
        features = K.variable(features)

        # call the Anchors layer
        anchors = anchors_layer.call(features)
        anchors = K.get_value(anchors)

        # expected anchor values
        expected = np.array([[
            [-12, -12, 20, 20],
            [-4, -12, 28, 20],
            [-12, -4, 20, 28],
            [-4, -4, 28, 28],
        ]], dtype=K.floatx())

        # test anchor values
        self.assertAllEqual(anchors, expected)

    def test_mini_batch(self):
        # create simple Anchors layer
        anchors_layer = layers.Anchors(
            size=32,
            stride=8,
            ratios=np.array([1], dtype=K.floatx()),
            scales=np.array([1], dtype=K.floatx()),
        )

        # create fake features input with batch_size=2
        features = np.zeros((2, 2, 2, 1024), dtype=K.floatx())
        features = K.variable(features)

        # call the Anchors layer
        anchors = anchors_layer.call(features)
        anchors = K.get_value(anchors)

        # expected anchor values
        expected = np.array([[
            [-12, -12, 20, 20],
            [-4, -12, 28, 20],
            [-12, -4, 20, 28],
            [-4, -4, 28, 28],
        ]], dtype=K.floatx())
        expected = np.tile(expected, (2, 1, 1))

        # test anchor values
        self.assertAllEqual(anchors, expected)


@keras_parameterized.run_all_keras_modes
class TestRegressBoxes(keras_parameterized.TestCase):

    def test_simple(self):
        # create simple RegressBoxes layer
        layer = layers.RegressBoxes()

        # create input
        anchors = np.array([[
            [0, 0, 10, 10],
            [50, 50, 100, 100],
            [20, 20, 40, 40],
        ]], dtype=K.floatx())
        anchors = K.variable(anchors)

        regression = np.array([[
            [0, 0, 0, 0],
            [0.1, 0.1, 0, 0],
            [0, 0, 0.1, 0.1],
        ]], dtype=K.floatx())
        regression = K.variable(regression)

        # compute output
        computed_shape = layer.compute_output_shape(
            [anchors.shape, regression.shape])
        actual = layer.call([anchors, regression])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, computed_shape)

        # compute expected output
        expected = np.array([[
            [0, 0, 10, 10],
            [51, 51, 100, 100],
            [20, 20, 40.4, 40.4],
        ]], dtype=K.floatx())

        self.assertAllClose(actual, expected)

    def test_mini_batch(self):
        mean = [0, 0, 0, 0]
        std = [0.2, 0.2, 0.2, 0.2]

        # create simple RegressBoxes layer
        layer = layers.RegressBoxes(mean=mean, std=std)

        # create input
        anchors = np.array([
            [
                [0, 0, 10, 10],  # 1
                [50, 50, 100, 100],  # 2
                [20, 20, 40, 40],  # 3
            ],
            [
                [20, 20, 40, 40],  # 3
                [0, 0, 10, 10],  # 1
                [50, 50, 100, 100],  # 2
            ],
        ], dtype=K.floatx())
        anchors = K.variable(anchors)

        regression = np.array([
            [
                [0, 0, 0, 0],  # 1
                [0.1, 0.1, 0, 0],  # 2
                [0, 0, 0.1, 0.1],  # 3
            ],
            [
                [0, 0, 0.1, 0.1],  # 3
                [0, 0, 0, 0],  # 1
                [0.1, 0.1, 0, 0],  # 2
            ],
        ], dtype=K.floatx())
        regression = K.variable(regression)

        # compute output
        actual = layer.call([anchors, regression])
        actual = K.get_value(actual)

        # compute expected output
        expected = np.array([
            [
                [0, 0, 10, 10],  # 1
                [51, 51, 100, 100],  # 2
                [20, 20, 40.4, 40.4],  # 3
            ],
            [
                [20, 20, 40.4, 40.4],  # 3
                [0, 0, 10, 10],  # 1
                [51, 51, 100, 100],  # 2
            ],
        ], dtype=K.floatx())

        self.assertAllClose(actual, expected)

    def test_invalid_input(self):
        bad_mean = 'invalid_data_type'
        bad_std = 'invalid_data_type'

        with self.assertRaises(ValueError):
            layers.RegressBoxes(mean=bad_mean, std=None)
        with self.assertRaises(ValueError):
            layers.RegressBoxes(mean=None, std=bad_std)


@keras_parameterized.run_all_keras_modes
class ClipBoxesTest(keras_parameterized.TestCase):

    def test_simple(self):
        img_h, img_w = np.random.randint(2, 5), np.random.randint(5, 9)

        boxes = np.array([[
            [9, 9, 9, 9],
            [-1, -1, -1, -1],
            [0, 0, img_w, img_h],
            [0, 0, img_w + 1, img_h + 1],
            [0, 0, img_w - 1, img_h - 1],
        ]], dtype='int')
        boxes = K.variable(boxes)

        # compute expected output
        expected = np.array([[
            [img_w - 1, img_h - 1, img_w - 1, img_h - 1],
            [0, 0, 0, 0],
            [0, 0, img_w - 1, img_h - 1],
            [0, 0, img_w - 1, img_h - 1],
            [0, 0, img_w - 1, img_h - 1],
        ]], dtype=K.floatx())

        # test channels_last
        # create input
        image = K.variable(np.random.random((1, img_h, img_w, 3)))

        # create simple ClipBoxes layer
        layer = layers.ClipBoxes(data_format='channels_last')

        # compute output
        computed_shape = layer.compute_output_shape(
            [image.shape, boxes.shape])
        actual = layer.call([image, boxes])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, tuple(computed_shape))
        self.assertAllClose(actual, expected)

        # test channels_first
        # create input
        image = K.variable(np.random.random((1, 6, img_h, img_w)))

        # create simple ClipBoxes layer
        layer = layers.ClipBoxes(data_format='channels_first')

        # compute output
        computed_shape = layer.compute_output_shape(
            [image.shape, boxes.shape])
        actual = layer.call([image, boxes])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, tuple(computed_shape))
        self.assertAllClose(actual, expected)

    def test_simple_3d(self):
        img_h, img_w = np.random.randint(2, 5), np.random.randint(5, 9)

        boxes = np.array([[
            [9, 9, 9, 9],
            [-1, -1, -1, -1],
            [0, 0, img_w, img_h],
            [0, 0, img_w + 1, img_h + 1],
            [0, 0, img_w - 1, img_h - 1],
        ]], dtype='int')
        boxes = np.expand_dims(boxes, axis=0)
        boxes = K.variable(boxes)

        # compute expected output
        expected = np.array([[
            [img_w - 1, img_h - 1, img_w - 1, img_h - 1],
            [0, 0, 0, 0],
            [0, 0, img_w - 1, img_h - 1],
            [0, 0, img_w - 1, img_h - 1],
            [0, 0, img_w - 1, img_h - 1],
        ]], dtype=K.floatx())
        expected = np.expand_dims(expected, axis=0)

        # test channels_last
        # create input
        image = K.variable(np.random.random((1, 1, img_h, img_w, 3)))

        # create simple ClipBoxes layer
        layer = layers.ClipBoxes(data_format='channels_last')

        # compute output
        computed_shape = layer.compute_output_shape(
            [image.shape, boxes.shape])
        actual = layer.call([image, boxes])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, tuple(computed_shape))
        self.assertAllClose(actual, expected)

        # test channels_first
        # create input
        image = K.variable(np.random.random((1, 6, 1, img_h, img_w)))

        # create simple ClipBoxes layer
        layer = layers.ClipBoxes(data_format='channels_first')

        # compute output
        computed_shape = layer.compute_output_shape(
            [image.shape, boxes.shape])
        actual = layer.call([image, boxes])
        actual = K.get_value(actual)

        self.assertEqual(actual.shape, tuple(computed_shape))
        self.assertAllClose(actual, expected)


if __name__ == '__main__':
    test.main()
