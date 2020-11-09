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
"""Test the RetinaNet models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.keras import backend as K
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.platform import test

from deepcell.model_zoo import RetinaNet


class RetinaNetTest(keras_parameterized.TestCase):

    @keras_parameterized.run_all_keras_modes
    @parameterized.named_parameters([
        {
            'testcase_name': 'retinanet_basic',
            'pooling': None,
            'panoptic': False,
            'location': False,
            'frames': 1,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_basic_td',
            'pooling': None,
            'panoptic': False,
            'location': False,
            'frames': 5,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_avgnorm',
            'pooling': 'avg',
            'panoptic': False,
            'location': False,
            'frames': 1,
            'pyramid_levels': ['P3', 'P4', 'P5'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_panoptic_maxnorm',
            'pooling': 'max',
            'panoptic': True,
            'location': False,
            'frames': 1,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_panoptic_maxnorm_td',
            'pooling': 'max',
            'panoptic': True,
            'location': False,
            'frames': 5,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_location',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 1,
            'pyramid_levels': ['P3', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_location_td',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 5,
            'pyramid_levels': ['P3', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_panoptic_location',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 1,
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_panoptic_location_td',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 5,
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
            'data_format': 'channels_last',
        },
        {
            'testcase_name': 'retinanet_basic_cf',
            'pooling': None,
            'panoptic': False,
            'location': False,
            'frames': 1,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_basic_cf_td',
            'pooling': None,
            'panoptic': False,
            'location': False,
            'frames': 5,
            'pyramid_levels': ['P3'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_avgnorm_cf',
            'pooling': 'avg',
            'panoptic': False,
            'location': False,
            'frames': 1,
            'pyramid_levels': ['P3', 'P4', 'P5'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_panoptic_maxnorm_cf',
            'pooling': 'max',
            'panoptic': True,
            'location': False,
            'frames': 1,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_panoptic_maxnorm_cf_td',
            'pooling': 'max',
            'panoptic': True,
            'location': False,
            'frames': 5,
            'pyramid_levels': ['P5', 'P6', 'P7'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_location_cf',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 1,
            'pyramid_levels': ['P3', 'P7'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_location_cf_td',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 5,
            'pyramid_levels': ['P3', 'P7'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_panoptic_location_cf',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 1,
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
            'data_format': 'channels_first',
        },
        {
            'testcase_name': 'retinanet_panoptic_location_cf_td',
            'pooling': 'max',
            'panoptic': True,
            'location': True,
            'frames': 5,
            'pyramid_levels': ['P3', 'P4', 'P5', 'P6', 'P7'],
            'data_format': 'channels_first',
        }
    ])
    def test_retinanet(self, pooling, panoptic, location,
                       frames, pyramid_levels, data_format):
        num_classes = 3
        norm_method = None

        # not all backbones work with channels_first
        backbone = 'featurenet'

        # TODO: TimeDistributed is incompatible with channels_first
        if frames > 1 and data_format == 'channels_first':
            return

        with self.cached_session():
            K.set_image_data_format(data_format)
            if data_format == 'channels_first':
                axis = 1
                input_shape = (1, 32, 32)
            else:
                axis = -1
                input_shape = (32, 32, 1)

            num_semantic_classes = [3, 4]
            if frames > 1:
                # TODO: 3D and semantic heads is not implemented.
                num_semantic_classes = []
            model = RetinaNet(
                backbone=backbone,
                num_classes=num_classes,
                input_shape=input_shape,
                norm_method=norm_method,
                location=location,
                pooling=pooling,
                panoptic=panoptic,
                frames_per_batch=frames,
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
