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
from tensorflow.python.platform import test

from deepcell import layers


@keras_parameterized.run_all_keras_modes
class TestFilterDetections(keras_parameterized.TestCase):

    def test_simple(self):
        # create simple FilterDetections layer
        layer = layers.FilterDetections()

        # create simple input
        boxes = np.array([[
            [0, 0, 10, 10],
            [0, 0, 10, 10],  # this will be suppressed
        ]], dtype=K.floatx())
        boxes = K.constant(boxes)

        classification = np.array([[
            [0, 0.9],  # this will be suppressed
            [0, 1],
        ]], dtype=K.floatx())
        classification = K.constant(classification)

        # compute output
        actual_boxes, actual_scores, actual_labels = layer.call(
            [boxes, classification])
        actual_boxes = K.get_value(actual_boxes)
        actual_scores = K.get_value(actual_scores)
        actual_labels = K.get_value(actual_labels)

        # define expected output
        expected_boxes = -1 * np.ones((1, 300, 4), dtype=K.floatx())
        expected_boxes[0, 0, :] = [0, 0, 10, 10]

        expected_scores = -1 * np.ones((1, 300), dtype=K.floatx())
        expected_scores[0, 0] = 1

        expected_labels = -1 * np.ones((1, 300), dtype=K.floatx())
        expected_labels[0, 0] = 1

        # assert actual and expected are equal
        self.assertAllEqual(actual_boxes, expected_boxes)
        self.assertAllEqual(actual_scores, expected_scores)
        self.assertAllEqual(actual_labels, expected_labels)

    def test_simple_3d(self):
        # create simple FilterDetections layer
        layer = layers.FilterDetections()

        # create simple input
        boxes = np.array([[
            [0, 0, 10, 10],
            [0, 0, 10, 10],  # this will be suppressed
        ]], dtype=K.floatx())
        boxes = np.expand_dims(boxes, 0)
        boxes = K.constant(boxes)

        classification = np.array([[
            [0, 0.9],  # this will be suppressed
            [0, 1],
        ]], dtype=K.floatx())
        classification = np.expand_dims(classification, 0)
        classification = K.constant(classification)

        # compute output
        actual_boxes, actual_scores, actual_labels = layer.call(
            [boxes, classification])
        actual_boxes = K.get_value(actual_boxes)
        actual_scores = K.get_value(actual_scores)
        actual_labels = K.get_value(actual_labels)

        # define expected output
        expected_boxes = -1 * np.ones((1, 1, 300, 4), dtype=K.floatx())
        expected_boxes[0, 0, 0, :] = [0, 0, 10, 10]

        expected_scores = -1 * np.ones((1, 1, 300), dtype=K.floatx())
        expected_scores[0, 0, 0] = 1

        expected_labels = -1 * np.ones((1, 1, 300), dtype=K.floatx())
        expected_labels[0, 0, 0] = 1

        # assert actual and expected are equal
        self.assertAllEqual(actual_boxes, expected_boxes)
        self.assertAllEqual(actual_scores, expected_scores)
        self.assertAllEqual(actual_labels, expected_labels)

    def test_simple_with_other(self):
        # create simple FilterDetections layer
        layer = layers.FilterDetections()

        # create simple input
        boxes = np.array([[
            [0, 0, 10, 10],
            [0, 0, 10, 10],  # this will be suppressed
        ]], dtype=K.floatx())
        boxes = K.constant(boxes)

        classification = np.array([[
            [0, 0.9],  # this will be suppressed
            [0, 1],
        ]], dtype=K.floatx())
        classification = K.constant(classification)

        other = []
        other.append(np.array([[
            [0, 1234],  # this will be suppressed
            [0, 5678],
        ]], dtype=K.floatx()))
        other.append(np.array([[
            5678,  # this will be suppressed
            1234,
        ]], dtype=K.floatx()))
        other = [K.constant(o) for o in other]

        # compute output
        actual = layer.call([boxes, classification] + other)
        actual_boxes = K.get_value(actual[0])
        actual_scores = K.get_value(actual[1])
        actual_labels = K.get_value(actual[2])
        actual_other = [K.get_value(a) for a in actual[3:]]

        # define expected output
        expected_boxes = -1 * np.ones((1, 300, 4), dtype=K.floatx())
        expected_boxes[0, 0, :] = [0, 0, 10, 10]

        expected_scores = -1 * np.ones((1, 300), dtype=K.floatx())
        expected_scores[0, 0] = 1

        expected_labels = -1 * np.ones((1, 300), dtype=K.floatx())
        expected_labels[0, 0] = 1

        expected_other = []
        expected_other.append(-1 * np.ones((1, 300, 2), dtype=K.floatx()))
        expected_other[-1][0, 0, :] = [0, 5678]
        expected_other.append(-1 * np.ones((1, 300), dtype=K.floatx()))
        expected_other[-1][0, 0] = 1234

        # assert actual and expected are equal
        self.assertAllEqual(actual_boxes, expected_boxes)
        self.assertAllEqual(actual_scores, expected_scores)
        self.assertAllEqual(actual_labels, expected_labels)

        for a, e in zip(actual_other, expected_other):
            self.assertAllEqual(a, e)

    def test_mini_batch(self):
        with self.cached_session():
            # create simple FilterDetections layer
            layer = layers.FilterDetections()

            # create input with batch_size=2
            boxes = np.array([
                [
                    [0, 0, 10, 10],  # this will be suppressed
                    [0, 0, 10, 10],
                ],
                [
                    [100, 100, 150, 150],
                    [100, 100, 150, 150],  # this will be suppressed
                ],
            ], dtype=K.floatx())
            boxes = K.constant(boxes)

            classification = np.array([
                [
                    [0, 0.9],  # this will be suppressed
                    [0, 1],
                ],
                [
                    [1, 0],
                    [0.9, 0],  # this will be suppressed
                ],
            ], dtype=K.floatx())
            classification = K.constant(classification)

            # compute output
            actual_boxes, actual_scores, actual_labels = layer.call(
                [boxes, classification])
            actual_boxes = K.get_value(actual_boxes)
            actual_scores = K.get_value(actual_scores)
            actual_labels = K.get_value(actual_labels)

            # define expected output
            expected_boxes = -1 * np.ones((2, 300, 4), dtype=K.floatx())
            expected_boxes[0, 0, :] = [0, 0, 10, 10]
            expected_boxes[1, 0, :] = [100, 100, 150, 150]

            expected_scores = -1 * np.ones((2, 300), dtype=K.floatx())
            expected_scores[0, 0] = 1
            expected_scores[1, 0] = 1

            expected_labels = -1 * np.ones((2, 300), dtype=K.floatx())
            expected_labels[0, 0] = 1
            expected_labels[1, 0] = 0

            # assert actual and expected are equal
            self.assertAllEqual(actual_boxes, expected_boxes)
            self.assertAllEqual(actual_scores, expected_scores)
            self.assertAllEqual(actual_labels, expected_labels)


if __name__ == '__main__':
    test.main()
