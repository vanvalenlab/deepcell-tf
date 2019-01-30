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
"""Tests for the retinanet layers"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell import layers


class TestAnchors(test.TestCase):
    def test_simple(self):
        with self.test_session():
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
            anchors = K.eval(anchors)

            # expected anchor values
            expected = np.array([[
                [-12, -12, 20, 20],
                [-4, -12, 28, 20],
                [-12, -4, 20, 28],
                [-4, -4, 28, 28],
            ]], dtype=K.floatx())

            # test anchor values
            self.assertAllEqual(anchors, expected)

    # mark test to fail
    def test_mini_batch(self):
        with self.test_session():
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
            anchors = K.eval(anchors)

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


class TestUpsampleLike(test.TestCase):
    def test_simple(self):
        with self.test_session():
            # create simple UpsampleLike layer
            upsample_like_layer = layers.UpsampleLike()

            # create input source
            source = np.zeros((1, 2, 2, 1), dtype=K.floatx())
            source = K.variable(source)
            target = np.zeros((1, 5, 5, 1), dtype=K.floatx())
            expected = target
            target = K.variable(target)

            # compute output
            actual = upsample_like_layer.call([source, target])
            actual = K.eval(actual)

            self.assertAllEqual(actual, expected)

    def test_mini_batch(self):
        with self.test_session():
            # create simple UpsampleLike layer
            upsample_like_layer = layers.UpsampleLike()

            # create input source
            source = np.zeros((2, 2, 2, 1), dtype=K.floatx())
            source = K.variable(source)

            target = np.zeros((2, 5, 5, 1), dtype=K.floatx())
            expected = target
            target = K.variable(target)

            # compute output
            actual = upsample_like_layer.call([source, target])
            actual = K.eval(actual)

            self.assertAllEqual(actual, expected)


class TestRegressBoxes(test.TestCase):
    def test_simple(self):
        with self.test_session():
            mean = [0, 0, 0, 0]
            std = [0.2, 0.2, 0.2, 0.2]

            # create simple RegressBoxes layer
            layer = layers.RegressBoxes(mean=mean, std=std)

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
            actual = layer.call([anchors, regression])
            actual = K.eval(actual)

            # compute expected output
            expected = np.array([[
                [0, 0, 10, 10],
                [51, 51, 100, 100],
                [20, 20, 40.4, 40.4],
            ]], dtype=K.floatx())

            self.assertAllClose(actual, expected)

    # mark test to fail
    def test_mini_batch(self):
        with self.test_session():
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
            actual = K.eval(actual)

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


# class AnchorsTest(test.TestCase):

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_anchors_2d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.Anchors,
#                 kwargs={'size': 1, 'stride': 1,
#                         'data_format': 'channels_last'},
#                 custom_objects={'Anchors': layers.Anchors},
#                 input_shape=(3, 5, 6, 4))
#             testing_utils.layer_test(
#                 layers.Anchors,
#                 kwargs={'inputs': 1, 'stride': 1,
#                         'data_format': 'channels_first'},
#                 custom_objects={'Anchors': layers.Anchors},
#                 input_shape=(3, 4, 5, 6))

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_anchors_3d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.Anchors,
#                 kwargs={'size': 1, 'stride': 1,
#                         'data_format': 'channels_last'},
#                 custom_objects={'Anchors': layers.Anchors},
#                 input_shape=(3, 11, 12, 10, 4))
#             testing_utils.layer_test(
#                 layers.Anchors,
#                 kwargs={'size': 1, 'stride': 1,
#                         'data_format': 'channels_first'},
#                 custom_objects={'Anchors': layers.Anchors},
#                 input_shape=(3, 4, 11, 12, 10))


# class UpsampleLikeTest(test.TestCase):

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_upsample_like_2d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.UpsampleLike,
#                 kwargs={'data_format': 'channels_last'},
#                 custom_objects={'UpsampleLike': layers.UpsampleLike},
#                 input_shape=(3, 5, 6, 4))
#             testing_utils.layer_test(
#                 layers.UpsampleLike,
#                 kwargs={'data_format': 'channels_first'},
#                 custom_objects={'UpsampleLike': layers.UpsampleLike},
#                 input_shape=(3, 4, 5, 6))

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_upsample_like_3d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.UpsampleLike,
#                 kwargs={'data_format': 'channels_last'},
#                 custom_objects={'UpsampleLike': layers.UpsampleLike},
#                 input_shape=(3, 11, 12, 10, 4))
#             testing_utils.layer_test(
#                 layers.UpsampleLike,
#                 kwargs={'data_format': 'channels_first'},
#                 custom_objects={'UpsampleLike': layers.UpsampleLike},
#                 input_shape=(3, 4, 11, 12, 10))


# class RegressBoxesTest(test.TestCase):

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_regress_boxes_2d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.RegressBoxes,
#                 kwargs={'data_format': 'channels_last'},
#                 custom_objects={'RegressBoxes': layers.RegressBoxes},
#                 input_shape=(3, 5, 6, 4))
#             testing_utils.layer_test(
#                 layers.RegressBoxes,
#                 kwargs={'data_format': 'channels_first'},
#                 custom_objects={'RegressBoxes': layers.RegressBoxes},
#                 input_shape=(3, 4, 5, 6))

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_regress_boxes_3d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.RegressBoxes,
#                 kwargs={'data_format': 'channels_last'},
#                 custom_objects={'RegressBoxes': layers.RegressBoxes},
#                 input_shape=(3, 11, 12, 10, 4))
#             testing_utils.layer_test(
#                 layers.RegressBoxes,
#                 kwargs={'data_format': 'channels_first'},
#                 custom_objects={'RegressBoxes': layers.RegressBoxes},
#                 input_shape=(3, 4, 11, 12, 10))


# class ClipBoxesTest(test.TestCase):

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_clip_boxes_2d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.ClipBoxes,
#                 kwargs={'data_format': 'channels_last'},
#                 custom_objects={'ClipBoxes': layers.ClipBoxes},
#                 input_shape=(3, 5, 6, 4))
#             testing_utils.layer_test(
#                 layers.ClipBoxes,
#                 kwargs={'data_format': 'channels_first'},
#                 custom_objects={'ClipBoxes': layers.ClipBoxes},
#                 input_shape=(3, 4, 5, 6))

#     @tf_test_util.run_in_graph_and_eager_modes()
#     def test_clip_boxes_3d(self):
#         with self.test_session(use_gpu=True):
#             testing_utils.layer_test(
#                 layers.ClipBoxes,
#                 kwargs={'data_format': 'channels_last'},
#                 custom_objects={'ClipBoxes': layers.ClipBoxes},
#                 input_shape=(3, 11, 12, 10, 4))
#             testing_utils.layer_test(
#                 layers.ClipBoxes,
#                 kwargs={'data_format': 'channels_first'},
#                 custom_objects={'ClipBoxes': layers.ClipBoxes},
#                 input_shape=(3, 4, 11, 12, 10))
