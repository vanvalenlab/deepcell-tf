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
"""Test the RetinaNet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.keras import backend as K

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test

from deepcell.model_zoo import RetinaNet


class RetinaNetTest(test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters([
        {
            'testcase_name': 'retinanet_basic',
            'pooling': None,
            'panoptic': False,
            'location': False,
            'pyramid_levels': ['P3'],
        },
        {
            'testcase_name': 'retinanet_avgnorm',
            'pooling': 'avg',
            'panoptic': False,
            'location': False,
            'pyramid_levels': ['P3', 'P4', 'P5'],
        },
        {
            'testcase_name': 'retinanet_panoptic_maxnorm',
            'pooling': 'max',
            'panoptic': True,
            'location': False,
            'pyramid_levels': ['P5', 'P6', 'P7'],
        },
        {
            'testcase_name': 'retinanet_location',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'pyramid_levels': ['P3', 'P7'],
        },
        {
            'testcase_name': 'retinanet_panoptic_location',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
        }
    ])
    # @tf_test_util.run_in_graph_and_eager_modes()
    def test_retinanet(self, pooling, panoptic, location, pyramid_levels):
        num_classes = 3
        norm_method = None
        backbone = 'featurenet'  # not all backbones work with channels_first

        for data_format in ('channels_last', 'channels_first'):
            with self.test_session(use_gpu=True):
                K.set_image_data_format(data_format)
                if data_format == 'channels_first':
                    axis = 1
                    input_shape = (1, 256, 256)
                else:
                    axis = -1
                    input_shape = (256, 256, 1)

                num_semantic_classes = [3, 4]
                model = RetinaNet(
                    backbone=backbone,
                    num_classes=num_classes,
                    input_shape=input_shape,
                    norm_method=norm_method,
                    location=location,
                    pooling=pooling,
                    panoptic=panoptic,
                    num_semantic_heads=len(num_semantic_classes),
                    num_semantic_classes=num_semantic_classes,
                    backbone_levels=['C3', 'C4', 'C5'],
                    pyramid_levels=pyramid_levels,
                )

                expected_size = 2 + panoptic * len(num_semantic_classes)
                self.assertIsInstance(model.output_shape, list)
                self.assertEqual(len(model.output_shape), expected_size)

                self.assertEqual(model.output_shape[0][-1], 4)
                self.assertEqual(model.output_shape[1][-1], num_classes)

                if panoptic:
                    for i, n in enumerate(num_semantic_classes):
                        self.assertEqual(model.output_shape[i + 2][axis], n)
